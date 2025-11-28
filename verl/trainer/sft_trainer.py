# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from functools import partial

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.metric.async_rollout_metrics import AsyncRolloutMetrics
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group
from verl.utils.flops_counter import FlopsCounter
from verl.utils.logger import log_with_rank
from verl.utils.tracking import Tracking
from verl.utils.dataset.rl_dataset import collate_fn

if is_cuda_available:
    pass
elif is_npu_available:
    pass

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        # Initialize resume-related variables
        self.resume_global_step = 0
        self.cumulative_tokens = 0  # Track cumulative tokens across all steps

        self._init_engine()

        self._build_ckpt_handler()

        self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        from verl.workers.roles.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        self.flops_counter = FlopsCounter(self.model_config.hf_config)

        self._build_rollout_metrics()

        if self.rank == 0:
            print(self.config)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
        )

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)

        self.rollout_url = getattr(self.config, "rollout_url", None)

    def _build_engine(self):
        from verl.workers.engine import BaseEngine, EngineRegistry

        self.engine: BaseEngine = EngineRegistry.new(
            model_type="language_model",
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

    def _init_engine(self):
        # patch optimizer config
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.engine.initialize()

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        
        
        if not tokenizer.chat_template:
            if self.rank == 0:
                print("WARNING: Tokenizer has no chat_template. Setting default ChatML template.")
            
            # This is the standard ChatML template (used by Qwen, newer models, etc)
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )
            
            # Ensure the tokenizer knows about these special tokens if they aren't standard
            # (Most tokenizers handle raw strings fine, but adding special tokens is safer)
            # if "<|im_start|>" not in tokenizer.all_special_tokens:
                # tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
        
        train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
        val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

        if hasattr(config.data, "rollout_files") and config.data.rollout_files is not None:
            if self.rank == 0:
                if hasattr(config.data, "rollout_max_size") and config.data.rollout_max_size is not None:
                    config.data.max_length = config.data.rollout_max_size

                config.data.add_generation_prompt = True
                rollout_dataset = create_sft_dataset(config.data.rollout_files, config.data, tokenizer)
            else:
                rollout_dataset = True
        else:
            rollout_dataset = None

        self.train_dataset, self.val_dataset, self.rollout_dataset = train_dataset, val_dataset, rollout_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        val_batch_size_per_dp = (config.data.val_batch_size or self.global_batch_size) // dp_size

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size_per_dp,
            sampler=self.val_sampler,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

    def _build_rollout_metrics(self):
        if self.rank == 0 and self.rollout_dataset is not None:
            self.rollout_metrics = AsyncRolloutMetrics(
                rollout_dataset=self.rollout_dataset,
                rollout_url=self.rollout_url,
                master_address=os.environ["MASTER_ADDR"],
                master_port=int(os.environ["MASTER_PORT"]) + 1,
                rollout_batch_size=self.config.data.rollout_batch_size,
                pad_token_id=3,  # TODO: fix this
            )
        else:
            self.rollout_metrics = None

    def _validate_rollout(self, is_logging, global_step, tracking, meta_info):
        last_valid_metric = None

        # Perform validation
        val_losses = []
        val_entropies = []
        for val_data in tqdm(self.val_dataloader, desc="validation", disable=not is_logging):
            total_tokens = val_data["response_mask"].sum().to(self.device_name)
            torch.distributed.all_reduce(
                total_tokens, op=torch.distributed.ReduceOp.SUM, group=self.engine.get_data_parallel_group()
            )
            # val_data.pop("rollout_params")
            with self.engine.eval_mode():
                # construct tensordict
                val_data = tu.get_tensordict(
                    tensor_dict=val_data,
                    non_tensor_dict={**meta_info, "calculate_entropy": True, "total_tokens": total_tokens.item()},
                )
                output = self.engine.infer_batch(data=val_data, loss_function=self.loss_fn)
                if self.engine.is_mp_src_rank_with_outputs():
                    response_mask = val_data["response_mask"].to(self.device_name).to(bool)
                    val_losses.append(output["loss"])
                    entropy = output["model_output"]["entropy"].to(self.device_name)
                    val_entropies.append(torch.sum(entropy * response_mask) / torch.sum(response_mask))

        if self.engine.is_mp_src_rank_with_outputs():
            val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
            val_entropy = torch.mean(torch.tensor(val_entropies, device=self.device_name))

            torch.distributed.all_reduce(
                val_loss, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()
            )
            torch.distributed.all_reduce(
                val_entropy, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()
            )

        if self.rollout_dataset:
            params = self.engine.get_state_dict_iter()

        if is_logging:
            metric = {
                "val/loss": val_loss.detach().item(),
                "val/entropy": val_entropy.detach().item(),
            }
            if self.rollout_metrics is not None:
                rollout_metrics, rollout_step = self.rollout_metrics.async_compute_metrics(global_step, params)
                print(f"{rollout_metrics=}")
                if rollout_metrics:
                    tracking.log(data=rollout_metrics, step=rollout_step)
            tracking.log(data=metric, step=global_step)
            last_valid_metric = metric
        torch.distributed.barrier()

        return last_valid_metric

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # TODO: add a unified tracking
        tracking = None
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "response_length": 256,
        }

        last_valid_metric = self._validate_rollout(is_logging, global_step, tracking, meta_info)

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # construct tensordict
                # data.pop("rollout_params")
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)

                total_tokens = data["response_mask"].sum().to(self.device_name)
                torch.distributed.all_reduce(
                    total_tokens, op=torch.distributed.ReduceOp.SUM, group=self.engine.get_data_parallel_group()
                )
                tu.assign_non_tensor(data, total_tokens=total_tokens.item())

                with self.engine.train_mode():
                    with Timer(name="update_policy", logger=None) as timer:
                        output = self.engine.train_batch(data=data, loss_function=self.loss_fn)
                lr = self.engine.lr_scheduler_step()

                if self.engine.is_mp_src_rank_with_outputs():
                    output_metrics = output["metrics"]

                    loss = torch.mean(torch.tensor(output_metrics["loss"], device=self.device_name))
                    torch.distributed.all_reduce(
                        loss, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()
                    )

                    batch_seqlens = data["attention_mask"].sum(dim=-1).to(self.device_name)
                    global_batch_size = batch_seqlens.shape[0] * self.engine.get_data_parallel_size()
                    global_batch_seqlens_tensor = torch.empty(
                        global_batch_size, dtype=batch_seqlens.dtype, device=self.device_name
                    )
                    torch.distributed.all_gather_into_tensor(
                        global_batch_seqlens_tensor, batch_seqlens, group=self.engine.get_data_parallel_group()
                    )
                    batch_seqlens = global_batch_seqlens_tensor.tolist()

                    self.cumulative_tokens += total_tokens.item()

                    metrics = {}
                    metrics["train/loss"] = loss.item()
                    metrics["train/grad_norm"] = output_metrics["grad_norm"]
                    metrics["train/lr"] = lr #.item()
                    metrics["train/global_tokens"] = total_tokens.item()
                    metrics["train/cumulative_tokens"] = self.cumulative_tokens
                    # mfu
                    delta_time = timer.last
                    estimated_flops, promised_flops = self.flops_counter.estimate_flops(batch_seqlens, delta_time)
                    metrics["train/mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()

                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.test_freq > 0 and is_valid_step):
                    last_valid_metric = self._validate_rollout(is_logging, global_step, tracking, meta_info)

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        if self.rollout_metrics is not None:
                            rollout_metrics, rollout_step = self.rollout_metrics.wait_compute_metrics()
                            if rollout_metrics:
                                tracking.log(data=rollout_metrics, step=rollout_step)

                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
