# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf
import time

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncRollouter,
)
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import (
    MessageQueue,
    MessageQueueClient,
)
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import (
    create_resource_pool_manager,
    create_role_worker_mapping,
)
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local
from dataclasses import dataclass
from typing import List

@dataclass
class FullyAsyncClusterConfig:
    trainer_head_id: str
    rollout_head_id: str
    trainer_nodes_ids: List[str]
    rollout_nodes_ids: List[str]

class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.node_ids = []
        self.train_node_ids = []
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self.cluster_config = validate_cluster(config)
        self._resolve_nodes()
        self._initialize_components(config)
        self._run_training_loop()

    def _resolve_nodes(self):
        self.fully_async_trainer_cls = ray.remote(
            num_cpus=10,
            label_selector={"actor": "true", "trainer_head": "true"}
        )(FullyAsyncTrainer).options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=self.cluster_config.trainer_head_id,
                soft=False,
            )
        )
        self.fully_async_rollout_cls = ray.remote(
            num_cpus=10,
            max_concurrency=100,
            label_selector={"rollout": "true", "rollout_head": "true"}
        )(FullyAsyncRollouter).options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=self.cluster_config.rollout_head_id,
                soft=False,
            )
        )

    def _initialize_components(self, config) -> None:
        print(
            f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing tokenizer...")
        use_shm = config.actor_rollout_ref.model.get("use_shm", False)
        tokenizer_path = (
            config.actor_rollout_ref.model.get("tokenizer_path")
            or config.actor_rollout_ref.model.path
        )
        local_tokenizer_path = copy_to_local(tokenizer_path, use_shm=use_shm)
        from verl.utils import hf_tokenizer_and_processor

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer, processor = hf_tokenizer_and_processor(
            local_tokenizer_path,
            trust_remote_code=trust_remote_code,
            processor_kwargs={"use_fast": True},
        )

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        self._create_trainer_rollouter(config)

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(
            self.components["trainer"].set_rollouter.remote(
                self.components["rollouter"]
            )
        )

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(
            self.components["rollouter"].get_total_train_steps.remote()
        )
        print(f"total_train_steps {total_train_steps}")
        ray.get(
            self.components["trainer"].set_total_train_steps.remote(total_train_steps)
        )

        # max_queue_size
        max_queue_size = ray.get(
            self.components["rollouter"].get_max_queue_size.remote()
        )
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get([
            self.components["rollouter"].set_message_queue_client.remote(
                self.components["message_queue_client"]
            ),
            self.components["trainer"].set_message_queue_client.remote(
                self.components["message_queue_client"]
            )
        ])

        # param_version resume from ckpt or default 0
        ray.get([
            self.components["trainer"].load_checkpoint.remote(), 
            self.components["rollouter"].load_checkpoint.remote()
        ])

        print("[ASYNC MAIN] Param sync before fit..")
        # This is the first iter of checkpoint_engine, meaning that no matter
        # what we load in the rollout it will be replaced by the trainer regardless
        # thus don't load rollout weights from disk, because it's useless
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _create_trainer_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer and rollouter...")
        start = time.perf_counter()
        rollouter = self.fully_async_rollout_cls.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = self.fully_async_trainer_cls.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(
                config, roles=list(trainer_role_mapping.keys())
            ),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get([trainer.init_workers.remote(), rollouter.init_workers.remote()])
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        self.components["trainer"] = trainer
        end = time.perf_counter()
        startup_time = (end-start)

        print(f"[ASYNC MAIN] Trainer and Rollouter created and initialized successfully in {startup_time} seconds")

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(
                    futures, num_returns=1, timeout=None
                )

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[ASYNC MAIN] Training completed or interrupted")

def validate_cluster(config):
    trainer_head_id = None
    rollout_head_id = None
    rollout_nodes = []
    trainer_nodes = []
    for node in ray.nodes():
        id = node["NodeID"]
        ip = node["NodeManagerAddress"]
        labels = node.get("Labels", {})
        assert "actor" in labels.keys() or "rollout" in labels.keys(), f"Node with id {id} and ip {ip} must be actor or rollout!" 
        if ("actor" in labels.keys()):
            assert labels["actor"] == "true", "Only valid value for 'actor' label is 'true'"
            trainer_nodes.append(id)
        if ("rollout" in labels.keys()):
            assert labels["rollout"] == "true", "Only valid value for 'rollout' label is 'true'"
            rollout_nodes.append(id)
        if ("trainer_head" in labels):
            assert "actor" in labels.keys(), "Cannot be trainer head and not a trainer node"
            assert labels["trainer_head"] == "true", "Only valid value for trainer_head label is 'true'"
            assert trainer_head_id == None, "Cannot have multiple trainer head nodes"
            trainer_head_id = node["NodeID"]
        if ("rollout_head" in labels):
            assert "rollout" in labels.keys(), "Cannot be rollout head and not a rollout node"
            assert labels["rollout_head"] == "true", "Only valid value for rollout_head label is 'true'"
            assert rollout_head_id == None, "Cannot have multiple rollout head nodes"
            rollout_head_id = node["NodeID"]

    assert config.rollout.nnodes == len(rollout_nodes), "Mismatch between nodes tagged rollout and rollout nodes in config"
    assert config.trainer.nnodes == len(trainer_nodes), "Mismatch between nodes tagged trainer and trainer nodes in config"
    assert trainer_head_id is not None, "No node specified as trainer head!"
    assert rollout_head_id is not None, "No node specified as rollout head!"
    return FullyAsyncClusterConfig(
        trainer_head_id=trainer_head_id,
        rollout_head_id=rollout_head_id,
        trainer_nodes_ids=trainer_nodes,
        rollout_nodes_ids=rollout_nodes
    )

@hydra.main(
    config_path="config", config_name="fully_async_ppo_trainer", version_base=None
)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(FullyAsyncTaskRunner))
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
