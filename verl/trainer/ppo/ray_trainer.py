# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.adaptive_rl_utils import (
    ADAPTIVE_KEEP_ALL_PAD_MASK_KEY,
    extract_reward_extra_info_row,
    finalize_prompt_rollouts,
    finalize_prompt_rollouts_keep_all,
    pad_adaptive_batch_to_divisor as pad_adaptive_dataproto_to_divisor,
    prune_prompt_caches,
    unpad_adaptive_batch as unpad_adaptive_dataproto_batch,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        reward_for_val: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            reward_for_val: Whether this is for validation
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If reward_for_val=False and sum_reward=True: summed reward_tensor (1D tensor)
            Otherwise: tuple of (reward_tensor, reward_extra_infos_dict)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if not reward_for_val and sum_reward:
                return reward_tensor

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = (
                {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
            )
            return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _get_filter_groups_config(self) -> dict[str, Any]:
        """Resolve filter-groups config from algorithm.filter_groups."""
        filter_cfg = self.config.algorithm.get("filter_groups", None) or {}
        max_num_gen_batches = filter_cfg.get("max_num_gen_batches", 0)
        if max_num_gen_batches is None:
            max_num_gen_batches = 0
        batch_target = filter_cfg.get("batch_target", "prompts")
        if batch_target is None:
            batch_target = "prompts"

        return {
            "enable": bool(filter_cfg.get("enable", False)),
            "metric": filter_cfg.get("metric", None),
            "max_num_gen_batches": int(max_num_gen_batches),
            "batch_target": str(batch_target).lower(),
        }

    def _validate_filter_groups_config(self, filter_cfg: dict[str, Any]) -> None:
        if not filter_cfg["enable"]:
            return

        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("Filter groups currently does not support algorithm.use_kl_in_reward=True.")
        if filter_cfg["metric"] is None:
            raise ValueError("algorithm.filter_groups.metric must be set when algorithm.filter_groups.enable=True.")
        if self.config.actor_rollout_ref.rollout.n <= 1:
            raise ValueError("Filter groups requires actor_rollout_ref.rollout.n > 1.")
        if filter_cfg["batch_target"] not in {"prompts", "samples"}:
            raise ValueError(
                "algorithm.filter_groups.batch_target must be one of {'prompts', 'samples'} "
                f"when algorithm.filter_groups.enable=True, got {filter_cfg['batch_target']}."
            )

    def _get_adaptive_group_sampling_config(self) -> dict[str, Any]:
        """Resolve adaptive group sampling config from algorithm.adaptive_group_sampling."""
        adaptive_cfg = self.config.algorithm.get("adaptive_group_sampling", None) or {}
        required_fields = (
            "enable",
            "min_positive_samples",
            "min_negative_samples",
            "max_rounds",
            "rollouts_per_round",
            "positive_threshold",
            "apply_downsampling",
            "apply_inverse_pass_rate_weight",
        )
        missing_fields = [field for field in required_fields if adaptive_cfg.get(field, None) is None]
        if missing_fields:
            raise KeyError(
                "Missing required adaptive_group_sampling config field(s): "
                f"{', '.join(missing_fields)}."
            )

        return {
            "enable": bool(adaptive_cfg["enable"]),
            "min_positive_samples": int(adaptive_cfg["min_positive_samples"]),
            "min_negative_samples": int(adaptive_cfg["min_negative_samples"]),
            "max_rounds": int(adaptive_cfg["max_rounds"]),
            "rollouts_per_round": int(adaptive_cfg["rollouts_per_round"]),
            "positive_threshold": float(adaptive_cfg["positive_threshold"]),
            "apply_downsampling": bool(adaptive_cfg["apply_downsampling"]),
            "apply_inverse_pass_rate_weight": bool(adaptive_cfg["apply_inverse_pass_rate_weight"]),
            "apply_prompt_inverse_group_weight": bool(adaptive_cfg.get("apply_prompt_inverse_group_weight", False)),
            "apply_within_prompt_mass_balance": bool(adaptive_cfg.get("apply_within_prompt_mass_balance", False)),
        }

    def _validate_adaptive_group_sampling_config(self, adaptive_cfg: dict[str, Any]) -> None:
        if not adaptive_cfg["enable"]:
            return

        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError(
                "algorithm.adaptive_group_sampling.enable=True is only supported with algorithm.adv_estimator=grpo."
            )
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError(
                "Adaptive group sampling currently does not support algorithm.use_kl_in_reward=True."
            )

        min_positive_samples = adaptive_cfg["min_positive_samples"]
        min_negative_samples = adaptive_cfg["min_negative_samples"]
        max_rounds = adaptive_cfg["max_rounds"]
        rollouts_per_round = adaptive_cfg["rollouts_per_round"]
        apply_downsampling = adaptive_cfg["apply_downsampling"]
        apply_prompt_inverse_group_weight = adaptive_cfg["apply_prompt_inverse_group_weight"]
        apply_within_prompt_mass_balance = adaptive_cfg["apply_within_prompt_mass_balance"]

        if min_positive_samples < 0 or min_negative_samples < 0:
            raise ValueError("Adaptive group sampling requires non-negative min_positive_samples/min_negative_samples.")
        if max_rounds <= 0:
            raise ValueError("Adaptive group sampling requires max_rounds > 0.")
        if rollouts_per_round <= 0:
            raise ValueError("Adaptive group sampling requires rollouts_per_round > 0.")
        if not isinstance(apply_downsampling, bool):
            raise ValueError("Adaptive group sampling requires apply_downsampling to be a boolean.")
        if not isinstance(apply_prompt_inverse_group_weight, bool):
            raise ValueError(
                "Adaptive group sampling requires apply_prompt_inverse_group_weight to be a boolean."
            )
        if not isinstance(apply_within_prompt_mass_balance, bool):
            raise ValueError(
                "Adaptive group sampling requires apply_within_prompt_mass_balance to be a boolean."
            )
        if (
            (not apply_downsampling)
            and self.config.trainer.balance_batch
            and bool(self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False))
        ):
            raise ValueError(
                "Adaptive keep-all mode (apply_downsampling=False) is currently incompatible with "
                "balance_batch=True when use_prefix_grouper=True. Please disable one of them."
            )

        target_rollouts = int(self.config.actor_rollout_ref.rollout.n)
        if apply_downsampling and (max_rounds * rollouts_per_round < target_rollouts):
            raise ValueError(
                "Adaptive group sampling requires max_rounds * rollouts_per_round >= actor_rollout_ref.rollout.n, "
                f"but got {max_rounds} * {rollouts_per_round} < {target_rollouts}."
            )

    def _generate_batch_with_adaptive_group_sampling(
        self,
        batch: DataProto,
        gen_batch: DataProto,
        adaptive_cfg: dict[str, Any],
        curr_step_profile: bool,
        timing_raw: dict[str, float],
        sleep_replicas_after_sampling: bool = True,
    ) -> tuple[DataProto, dict[str, list[Any]], dict[str, float]]:
        """Generate adaptive-group rollouts and build the training batch."""
        target_rollouts = int(self.config.actor_rollout_ref.rollout.n)  # NOTE: target_rollouts is only enforced when downsampling
        rollouts_per_round = int(adaptive_cfg["rollouts_per_round"])
        max_rounds = int(adaptive_cfg["max_rounds"])
        min_positive_samples = int(adaptive_cfg["min_positive_samples"])
        min_negative_samples = int(adaptive_cfg["min_negative_samples"])
        positive_threshold = float(adaptive_cfg["positive_threshold"])
        apply_downsampling = bool(adaptive_cfg["apply_downsampling"])

        num_prompts = len(batch)
        prompt_uids = [str(uid) for uid in batch.non_tensor_batch["uid"].tolist()]
        if len(set(prompt_uids)) != num_prompts:
            raise ValueError("Adaptive group sampling requires unique prompt uids per prompt.")
        # Route sampled rollouts back to prompt states by uid
        prompt_uid_to_idx = {uid: idx for idx, uid in enumerate(prompt_uids)}
        prompt_states: list[dict[str, Any]] = []
        for _ in range(num_prompts):
            prompt_states.append(
                {
                    # Running counts/statistics computed with ALL sampled rollouts
                    "total": 0,
                    "pos": 0,
                    "neg": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    # Per-class cached rollout payloads used during final selection
                    "positive_cache": [],
                    "negative_cache": [],
                    # Final payloads that will be concatenated into the training DataProto
                    "selected": [],
                    "selected_pos": 0,
                    "selected_neg": 0,
                    "finalized": False,
                }
            )

        # Prompt is active while it still needs additional sampled rollouts
        active_mask = np.ones(num_prompts, dtype=bool)
        prompts_completed_early = 0
        rounds_executed = 0

        for i in range(max_rounds):
            print(
                f"[Adaptive Sampling] Round {i+1}/{max_rounds} - Executing rollouts for active prompts ({active_mask.sum()})"
            )

            active_indices = np.where(active_mask)[0]
            if active_indices.size == 0:
                break
            rounds_executed += 1

            active_prompt_batch = batch.select_idxs(active_indices.tolist())
            active_gen_batch = gen_batch.select_idxs(active_indices.tolist())

            round_prompt_batch = active_prompt_batch.repeat(repeat_times=rollouts_per_round, interleave=True)
            round_gen_batch = active_gen_batch.repeat(repeat_times=rollouts_per_round, interleave=True)
            round_gen_batch.meta_info["global_steps"] = self.global_steps
            # Rollout backend requires divisibility by worker/rank split factor
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            round_gen_batch_padded, round_pad_size = pad_dataproto_to_divisor(round_gen_batch, size_divisor)

            if not self.async_rollout_mode:
                round_gen_output_padded = self.actor_rollout_wg.generate_sequences(round_gen_batch_padded)
            else:
                if curr_step_profile:
                    self.async_rollout_manager.start_profile(global_step=self.global_steps)
                round_gen_output_padded = self.async_rollout_manager.generate_sequences(round_gen_batch_padded)
                if curr_step_profile:
                    self.async_rollout_manager.stop_profile()
            round_gen_output = unpad_dataproto(round_gen_output_padded, pad_size=round_pad_size)

            round_timing = round_gen_output.meta_info.get("timing", {})
            for key, value in round_timing.items():
                timing_raw[key] = timing_raw.get(key, 0.0) + value
            round_gen_output.meta_info.pop("timing", None)

            # Union prompt tensors and generated response tensors for reward computation
            round_batch = round_prompt_batch.union(round_gen_output)

            # Reward path mirrors regular trainer logic, but done round-by-round
            if self.use_rm and "rm_scores" not in round_batch.batch.keys():
                if not self.use_reward_loop:
                    round_rm_scores = self.rm_wg.compute_rm_score(round_batch)
                else:
                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                    round_rm_scores = self.reward_loop_manager.compute_rm_score(round_batch)
                round_batch = round_batch.union(round_rm_scores)

            if self.config.reward_model.launch_reward_fn_async:
                round_reward_tensor, round_reward_extra_infos_dict = ray.get(
                    compute_reward_async.remote(data=round_batch, config=self.config, tokenizer=self.tokenizer)
                )
            else:
                round_reward_tensor, round_reward_extra_infos_dict = self._compute_or_extract_reward(
                    round_batch, reward_fn=self.reward_fn, reward_for_val=False
                )

            sequence_scores = round_reward_tensor.sum(dim=-1).detach().cpu().tolist()
            round_batch_size = len(round_batch)

            for sample_idx, score in enumerate(sequence_scores):
                sample_uid = str(round_batch.non_tensor_batch["uid"][sample_idx])
                if sample_uid not in prompt_uid_to_idx:
                    raise ValueError(f"Unknown prompt uid {sample_uid} encountered during adaptive sampling.")
                prompt_idx = prompt_uid_to_idx[sample_uid]
                prompt_state = prompt_states[prompt_idx]

                score_value = float(score)
                is_positive = score_value > positive_threshold
                prompt_state["total"] += 1
                prompt_state["sum"] += score_value
                prompt_state["sumsq"] += score_value * score_value
                if is_positive:
                    prompt_state["pos"] += 1
                else:
                    prompt_state["neg"] += 1

                # Build one-sample DataProto payload to keep only what PPO update needs
                sample = round_batch.select_idxs([sample_idx])
                sample.batch["token_level_scores"] = round_reward_tensor[sample_idx : sample_idx + 1]
                sample_extra = extract_reward_extra_info_row(
                    round_reward_extra_infos_dict, sample_idx, round_batch_size
                )
                sample_entry = {
                    "sample": sample,
                    "extra": sample_extra,
                    "is_positive": is_positive,
                }

                if is_positive:
                    prompt_state["positive_cache"].append(sample_entry)
                else:
                    prompt_state["negative_cache"].append(sample_entry)

                # Prune cached samples when downsampling is ON
                if apply_downsampling:
                    prune_prompt_caches(prompt_state, target_rollouts=target_rollouts)

            # Prompt-level stopping/finalization check after processing current round
            for prompt_idx in active_indices:
                prompt_state = prompt_states[prompt_idx]
                if prompt_state["finalized"]:
                    continue
                meets_class_criteria = (
                    prompt_state["pos"] >= min_positive_samples and prompt_state["neg"] >= min_negative_samples
                )
                if meets_class_criteria and (
                    (apply_downsampling and prompt_state["total"] >= target_rollouts) or (not apply_downsampling)
                ):
                    if apply_downsampling:
                        finalize_prompt_rollouts(prompt_state, target_rollouts=target_rollouts)
                    else:
                        finalize_prompt_rollouts_keep_all(prompt_state)
                    active_mask[prompt_idx] = False
                    prompts_completed_early += 1

        # Prompts still active after all rounds are exactly those that hit max_rounds
        # without satisfying the early-finalization constraints.
        active_at_max_rounds = np.where(active_mask)[0] if rounds_executed >= max_rounds else np.array([], dtype=int)
        active_unmet_min_positive = 0
        active_unmet_min_negative = 0
        for prompt_idx in active_at_max_rounds:
            prompt_state = prompt_states[int(prompt_idx)]
            if prompt_state["pos"] < min_positive_samples:
                active_unmet_min_positive += 1
            if prompt_state["neg"] < min_negative_samples:
                active_unmet_min_negative += 1

        # Finalize unresolved prompts with the best available balanced subset.
        for prompt_state in prompt_states:
            if prompt_state["finalized"]:
                continue
            if apply_downsampling and (prompt_state["total"] < target_rollouts):
                raise ValueError(
                    f"Prompt sampled only {prompt_state['total']} rollouts; expected at least {target_rollouts}. "
                    "Please increase max_rounds or rollouts_per_round."
                )
            if apply_downsampling:
                finalize_prompt_rollouts(prompt_state, target_rollouts=target_rollouts)
            else:
                finalize_prompt_rollouts_keep_all(prompt_state)

        # Keep rollout KV cache between rounds and sleep only once after adaptive sampling.
        if self.async_rollout_mode and sleep_replicas_after_sampling:
            self.checkpoint_manager.sleep_replicas()

        # Materialize selected samples and attach per-prompt statistics
        selected_samples: list[DataProto] = []
        reward_extra_infos_dict: dict[str, list[Any]] = defaultdict(list)
        rollouts_per_prompt = []
        pass_rates = []
        selected_pos = []
        selected_neg = []

        for prompt_state in prompt_states:
            total = int(prompt_state["total"])
            mean = float(prompt_state["sum"] / total)
            var = max(float(prompt_state["sumsq"] / total - mean * mean), 0.0)
            std = float(var**0.5)
            pass_rate = float(prompt_state["pos"] / total)

            rollouts_per_prompt.append(total)
            pass_rates.append(pass_rate)
            selected_pos.append(int(prompt_state["selected_pos"]))
            selected_neg.append(int(prompt_state["selected_neg"]))

            for entry in prompt_state["selected"]:
                sample = entry["sample"]
                score_dtype = sample.batch["token_level_scores"].dtype
                score_device = sample.batch["token_level_scores"].device
                sample.batch["adaptive_group_mean"] = torch.tensor([mean], dtype=score_dtype, device=score_device)
                sample.batch["adaptive_group_std"] = torch.tensor([std], dtype=score_dtype, device=score_device)
                sample.batch["adaptive_group_pass_rate"] = torch.tensor(
                    [pass_rate], dtype=score_dtype, device=score_device
                )
                selected_samples.append(sample)
                for key, value in entry["extra"].items():
                    reward_extra_infos_dict[key].append(value)

        if apply_downsampling:
            expected_batch_size = num_prompts * target_rollouts
            if len(selected_samples) != expected_batch_size:
                raise ValueError(
                    f"Adaptive group sampling built {len(selected_samples)} selected samples, "
                    f"expected {expected_batch_size}."
                )
        else:
            if len(selected_samples) == 0:
                raise ValueError("Adaptive keep-all sampling produced zero selected samples.")
            for prompt_idx, prompt_state in enumerate(prompt_states):
                selected_count = len(prompt_state["selected"])
                total_count = int(prompt_state["total"])
                if selected_count != total_count:
                    raise ValueError(
                        "Adaptive keep-all sampling invariant violated: "
                        f"prompt_idx={prompt_idx} selected_count={selected_count} total_count={total_count}."
                    )

        selected_batch = DataProto.concat(selected_samples)

        selected_pos_arr = np.array(selected_pos, dtype=np.float32)
        selected_neg_arr = np.array(selected_neg, dtype=np.float32)
        metrics = {
            "adaptive_group_sampling/total_rollouts": float(np.sum(rollouts_per_prompt)),
            "adaptive_group_sampling/rollouts_per_prompt_mean": float(np.mean(rollouts_per_prompt)),
            "adaptive_group_sampling/rollouts_per_prompt_min": float(np.min(rollouts_per_prompt)),
            "adaptive_group_sampling/rollouts_per_prompt_max": float(np.max(rollouts_per_prompt)),
            "adaptive_group_sampling/rounds_executed": float(rounds_executed),
            "adaptive_group_sampling/prompts_not_completed_in_rounds": float(num_prompts - prompts_completed_early),
            "adaptive_group_sampling/max_rounds_unmet_pos_min": float(active_unmet_min_positive),
            "adaptive_group_sampling/max_rounds_unmet_neg_min": float(active_unmet_min_negative),
            # NOTE: this is computed with all rollouts, while critic/rewards/mean only sees the selected rollouts
            "adaptive_group_sampling/pass_rate_mean": float(np.mean(pass_rates)),
            "adaptive_group_sampling/selected_positive_per_prompt_mean": float(np.mean(selected_pos_arr)),
            "adaptive_group_sampling/selected_negative_per_prompt_mean": float(np.mean(selected_neg_arr)),
            "adaptive_group_sampling/selected_positive_ratio_mean": float(
                np.mean(selected_pos_arr / (selected_pos_arr + selected_neg_arr + 1e-8))
            ),
        }
        return selected_batch, dict(reward_extra_infos_dict), metrics

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            rm_resource_pool=rm_resource_pool,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _get_training_dp_divisor(self) -> int:
        """Compute a common size divisor across actor/ref/critic DP meshes for one train step."""

        def resolve_dp_size(worker_group, role_candidates: tuple[str, ...]) -> int:
            last_error = None
            for role in role_candidates:
                try:
                    return self._get_dp_size(worker_group, role)
                except (KeyError, ValueError, AssertionError, AttributeError, IndexError, RuntimeError) as exc:
                    last_error = exc
            raise RuntimeError(
                f"Failed to resolve dp_size for worker_group with roles={role_candidates}. "
                f"Last error: {type(last_error).__name__}: {last_error}"
            )

        divisors = [resolve_dp_size(self.actor_rollout_wg, ("actor", "train"))]

        if self.use_reference_policy and not self.ref_in_actor:
            divisors.append(resolve_dp_size(self.ref_policy_wg, ("ref", "actor", "train")))

        if self.use_critic:
            divisors.append(resolve_dp_size(self.critic_wg, ("critic", "train", "actor")))

        size_divisor = 1
        for dp_size in divisors:
            size_divisor = math.lcm(size_divisor, dp_size)
        return size_divisor

    def pad_adaptive_batch_to_divisor(self, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Pad adaptive batch to the common DP divisor and record padding metrics."""
        metrics.setdefault("adaptive_group_sampling/dp_padding_added", 0.0)
        metrics.setdefault("adaptive_group_sampling/dp_padding_ratio", 0.0)
        metrics.setdefault("adaptive_group_sampling/balance_applied_keep_all", 0.0)

        original_size = len(batch)
        size_divisor = self._get_training_dp_divisor()
        batch, pad_size = pad_adaptive_dataproto_to_divisor(
            batch=batch,
            size_divisor=size_divisor,
            mask_key=ADAPTIVE_KEEP_ALL_PAD_MASK_KEY,
        )
        metrics["adaptive_group_sampling/dp_padding_added"] = float(pad_size)
        metrics["adaptive_group_sampling/dp_padding_ratio"] = float(pad_size / max(original_size, 1))
        return batch

    def unpad_adaptive_batch(self, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Remove adaptive padding rows and validate that removal count matches tracked metrics."""
        expected_pad = int(metrics.get("adaptive_group_sampling/dp_padding_added", 0.0))
        batch, removed_pad = unpad_adaptive_dataproto_batch(
            batch=batch,
            mask_key=ADAPTIVE_KEEP_ALL_PAD_MASK_KEY,
        )
        if removed_pad != expected_pad:
            raise ValueError(
                "Adaptive keep-all padding mismatch: "
                f"expected_pad={expected_pad}, removed_pad={removed_pad}."
            )
        if "attention_mask" in batch.batch.keys():
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        return batch

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        try:
            # Use group-level balancing for PrefixGrouper to keep same-uid samples together.
            if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
                from verl.utils.seqlen_balancing import get_group_balanced_partitions

                uid_list = list(batch.non_tensor_batch["uid"])
                seqlen_list = global_seqlen_lst.tolist()

                # Count number of uid groups
                num_groups = len(set(uid_list))

                if num_groups % dp_size != 0:
                    raise ValueError(
                        f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                        f"% dp_size ({dp_size}) == 0. "
                        f"This ensures each rank gets equal number of groups. "
                        f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                        f"dp_size * rollout.n."
                    )

                global_partition_lst = get_group_balanced_partitions(
                    seqlen_list=seqlen_list,
                    uid_list=uid_list,
                    k_partitions=dp_size,
                )

            elif keep_minibatch:
                # Decouple the DP balancing and mini-batching.
                minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
                minibatch_num = len(workload_lst) // minibatch_size
                global_partition_lst = [[] for _ in range(dp_size)]
                for i in range(minibatch_num):
                    rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                        workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                        k_partitions=dp_size,
                        equal_size=True,
                    )
                    for j, part in enumerate(rearrange_minibatch_lst):
                        global_partition_lst[j].extend([x + minibatch_size * i for x in part])
            else:
                global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        except (AssertionError, ValueError):
            raise

        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        if ADAPTIVE_KEEP_ALL_PAD_MASK_KEY in batch.non_tensor_batch:
            metrics["adaptive_group_sampling/balance_applied_keep_all"] = 1.0

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            # TODO: (adaptive-keep-all) This scaling still depends on rollout.n.
            # When adaptive mode is ON without downsampling, the effective train batch size can vary across steps,
            # so a dynamic mini/global batch sizing policy may be more appropriate.
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            # TODO: (adaptive-keep-all) Same coupling as actor path - rollout.n is used to
            # scale critic mini-batch size even when adaptive keep-all changes actual sample count.
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        adaptive_group_sampling_cfg = self._get_adaptive_group_sampling_config()
        self._validate_adaptive_group_sampling_config(adaptive_group_sampling_cfg)
        filter_groups_cfg = self._get_filter_groups_config()
        self._validate_filter_groups_config(filter_groups_cfg)

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        pending_filtered_batch = None
        pending_prompt_count = 0
        pending_sample_count = 0
        pending_num_gen_batches = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                adaptive_group_sampling_enabled = bool(adaptive_group_sampling_cfg["enable"])
                filter_groups_enabled = bool(filter_groups_cfg["enable"])
                adaptive_keep_all_enabled = adaptive_group_sampling_enabled and (
                    not bool(adaptive_group_sampling_cfg["apply_downsampling"])
                )
                reward_extra_infos_dict: dict[str, list[Any]] = {}
                reward_precomputed = False
                future_reward = None
                rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                prompt_bsz = int(self.config.data.train_batch_size)
                sample_bsz = prompt_bsz * rollout_n

                if filter_groups_enabled:
                    # Optimization when filter-groups is enabled: only generate the number of prompts
                    # needed to satisfy the current accumulation target.
                    if filter_groups_cfg["batch_target"] == "samples":
                        missing_samples = max(sample_bsz - pending_sample_count, 0)
                        if adaptive_group_sampling_enabled and (not adaptive_group_sampling_cfg["apply_downsampling"]):
                            missing_prompts = missing_samples
                        else:
                            missing_prompts = int(math.ceil(missing_samples / max(rollout_n, 1)))
                    else:
                        missing_prompts = max(prompt_bsz - pending_prompt_count, 0)
                    missing_prompts = max(1, min(len(batch), int(missing_prompts)))
                    if missing_prompts < len(batch):
                        batch = batch.select_idxs(list(range(missing_prompts)))
                        gen_batch = gen_batch.select_idxs(list(range(missing_prompts)))
                gen_batch_output = None

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if adaptive_group_sampling_enabled:
                            batch, reward_extra_infos_dict, adaptive_sampling_metrics = (
                                self._generate_batch_with_adaptive_group_sampling(
                                    batch=batch,
                                    gen_batch=gen_batch,
                                    adaptive_cfg=adaptive_group_sampling_cfg,
                                    curr_step_profile=curr_step_profile,
                                    timing_raw=timing_raw,
                                    sleep_replicas_after_sampling=False,
                                )
                            )
                            metrics.update(adaptive_sampling_metrics)
                        else:
                            gen_batch_output = gen_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile(global_step=self.global_steps)
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                    if (not adaptive_group_sampling_enabled) and self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    if not adaptive_group_sampling_enabled:
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                    elif reward_extra_infos_dict:
                        adaptive_reward_extras = {}
                        for key, values in reward_extra_infos_dict.items():
                            values_arr = np.array(values)
                            if values_arr.ndim > 0 and values_arr.shape[0] != len(batch):
                                raise ValueError(
                                    "Adaptive reward extra info length mismatch before balancing: "
                                    f"key={key}, value_len={values_arr.shape[0]}, batch_len={len(batch)}."
                                )
                            adaptive_reward_extras[key] = values_arr
                        batch.non_tensor_batch.update(adaptive_reward_extras)

                    # Compute rewards before filter-groups so we can early-continue without
                    # running old_log_prob/ref/value/adv on under-filled updates.
                    if not reward_precomputed:
                        with marked_timer("reward", timing_raw, color="yellow"):
                            if not adaptive_group_sampling_enabled:
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    if not self.use_reward_loop:
                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                    else:
                                        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                        reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                                    batch = batch.union(reward_tensor)

                                if filter_groups_enabled:
                                    reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                        batch, reward_fn=self.reward_fn, reward_for_val=False
                                    )
                                    batch.batch["token_level_scores"] = reward_tensor
                                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                                    if reward_extra_infos_dict:
                                        batch.non_tensor_batch.update(
                                            {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                        )
                                elif self.config.reward_model.launch_reward_fn_async:
                                    future_reward = compute_reward_async.remote(
                                        data=batch, config=self.config, tokenizer=self.tokenizer
                                    )
                                else:
                                    reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                        batch, reward_fn=self.reward_fn, reward_for_val=False
                                    )

                    # Filter out groups of samples if configured, based on reward metric variance.
                    if filter_groups_enabled:
                        pending_num_gen_batches += 1

                        metric_name = filter_groups_cfg["metric"]
                        if metric_name == "seq_final_reward":
                            seq_reward_tensor = (
                                batch.batch["token_level_rewards"]
                                if "token_level_rewards" in batch.batch
                                else batch.batch["token_level_scores"]
                            )
                            batch.non_tensor_batch["seq_final_reward"] = (
                                seq_reward_tensor.sum(dim=-1).detach().cpu().numpy()
                            )
                        elif metric_name == "seq_reward":
                            batch.non_tensor_batch["seq_reward"] = (
                                batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
                            )
                        elif metric_name not in batch.non_tensor_batch:
                            raise KeyError(
                                f"Filter metric '{metric_name}' not found in batch.non_tensor_batch. "
                                "Expected one of reward extra-info keys or seq_reward/seq_final_reward."
                            )

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        kept_prompt_uids = [
                            uid
                            for uid, metric_vals in prompt_uid2metric_vals.items()
                            if np.std(metric_vals) > 0 or len(metric_vals) == 1
                        ]
                        kept_prompt_count_this_round = len(kept_prompt_uids)
                        pending_prompt_count += kept_prompt_count_this_round

                        kept_traj_idxs = [
                            idx
                            for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"])
                            if traj_from_prompt_uid in kept_prompt_uids
                        ]
                        kept_sample_count_this_round = len(kept_traj_idxs)
                        pending_sample_count += kept_sample_count_this_round
                        batch = batch[kept_traj_idxs]
                        pending_filtered_batch = (
                            batch
                            if pending_filtered_batch is None
                            else DataProto.concat([pending_filtered_batch, batch])
                        )

                        batch_target = filter_groups_cfg["batch_target"]
                        if batch_target == "samples":
                            has_enough = pending_sample_count >= sample_bsz
                            curr_count = pending_sample_count
                            target_count = sample_bsz
                        else:
                            has_enough = pending_prompt_count >= prompt_bsz
                            curr_count = pending_prompt_count
                            target_count = prompt_bsz

                        if not has_enough:
                            print(
                                "[filter_groups] criteria not met; "
                                f"round={pending_num_gen_batches}, "
                                f"kept_prompts_this_round={kept_prompt_count_this_round}, "
                                f"kept_samples_this_round={kept_sample_count_this_round}, "
                                f"accumulated_{batch_target}={curr_count}/{target_count}"
                            )
                            max_num_gen_batches = int(filter_groups_cfg["max_num_gen_batches"])
                            if max_num_gen_batches <= 0 or pending_num_gen_batches < max_num_gen_batches:
                                print(f"[filter_groups] keep generating (rounds_so_far={pending_num_gen_batches})")
                                continue
                            raise ValueError(
                                f"{pending_num_gen_batches=} >= {max_num_gen_batches=}."
                                + " Generated too many. Please check if your data are too difficult."
                                + " You could also try set max_num_gen_batches=0 to enable endless trials."
                            )
                        elif pending_num_gen_batches > 1:
                            print(
                                "[filter_groups] criteria met; "
                                f"rounds_executed={pending_num_gen_batches}, "
                                f"accumulated_{batch_target}={curr_count}/{target_count}"
                            )

                        if pending_filtered_batch is None:
                            raise ValueError("No filtered samples collected. Please check filter_groups settings.")
                        prompt_uid2sample_count = defaultdict(int)
                        for prompt_uid in pending_filtered_batch.non_tensor_batch["uid"]:
                            prompt_uid2sample_count[prompt_uid] += 1

                        selected_prompt_count = 0
                        selected_sample_count = 0
                        selected_prompt_uid_set = set()
                        for prompt_uid in pending_filtered_batch.non_tensor_batch["uid"]:
                            if prompt_uid in selected_prompt_uid_set:
                                continue
                            selected_prompt_uid_set.add(prompt_uid)
                            selected_prompt_count += 1
                            selected_sample_count += prompt_uid2sample_count[prompt_uid]
                            if (
                                (batch_target == "samples" and selected_sample_count >= sample_bsz)
                                or (batch_target == "prompts" and selected_prompt_count >= prompt_bsz)
                            ):
                                break
                        selected_traj_idxs = [
                            idx
                            for idx, traj_from_prompt_uid in enumerate(pending_filtered_batch.non_tensor_batch["uid"])
                            if traj_from_prompt_uid in selected_prompt_uid_set
                        ]
                        batch = pending_filtered_batch[selected_traj_idxs]
                        reward_precomputed = True
                        reward_extra_infos_dict = {}
                        metrics["train/num_gen_batches"] = float(pending_num_gen_batches)

                    # Async rollout replicas should only be slept when generation for this
                    # training step is finalized (i.e., not continuing filter retry rounds).
                    if self.async_rollout_mode:
                        self.checkpoint_manager.sleep_replicas()

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if adaptive_keep_all_enabled:
                        batch = self.pad_adaptive_batch_to_divisor(batch=batch, metrics=metrics)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    if "multi_modal_inputs" in batch.non_tensor_batch:
                        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                            if "image_grid_thw" not in multi_modal_input.keys():
                                continue
                            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if adaptive_group_sampling_enabled:
                            reward_tensor = batch.batch["token_level_scores"]
                        elif not reward_precomputed:
                            # we combine with rule-based rm
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict and (not adaptive_group_sampling_enabled) and (not reward_precomputed):
                            # NOTE: When adaptive sampling is enabled, the reward extra infos are already added to batch.non_tensor_batch
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor
                        if adaptive_group_sampling_enabled:
                            advantages, returns, adaptive_weights = core_algos.compute_adaptive_grpo_outcome_advantage(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                response_mask=batch.batch["response_mask"],
                                group_mean=batch.batch["adaptive_group_mean"],
                                group_std=batch.batch["adaptive_group_std"],
                                group_pass_rate=batch.batch["adaptive_group_pass_rate"],
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                apply_inverse_pass_rate_weight=adaptive_group_sampling_cfg[
                                    "apply_inverse_pass_rate_weight"
                                ],
                            )
                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = returns
                            if adaptive_group_sampling_cfg["apply_prompt_inverse_group_weight"] or adaptive_group_sampling_cfg[
                                "apply_within_prompt_mass_balance"
                            ]:
                                batch.batch["advantages"] = core_algos.apply_adaptive_prompt_advantage_weighting(
                                    advantages=batch.batch["advantages"],
                                    response_mask=batch.batch["response_mask"],
                                    index=batch.non_tensor_batch["uid"],
                                    apply_prompt_inverse_group_weight=adaptive_group_sampling_cfg[
                                        "apply_prompt_inverse_group_weight"
                                    ],
                                    apply_within_prompt_mass_balance=adaptive_group_sampling_cfg[
                                        "apply_within_prompt_mass_balance"
                                    ],
                                )
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if adaptive_keep_all_enabled:
                        batch = self.unpad_adaptive_batch(batch=batch, metrics=metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        # sleep replicas to avoid OOM during checkpoint saving
                        self.checkpoint_manager.sleep_replicas()
                        self._save_checkpoint()
                        # wake replicas to avoid OOM during checkpoint saving
                        self.checkpoint_manager.update_weights()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # reset pending batch and filtering state
                if filter_groups_enabled:
                    pending_filtered_batch = None
                    pending_prompt_count = 0
                    pending_sample_count = 0
                    pending_num_gen_batches = 0

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
