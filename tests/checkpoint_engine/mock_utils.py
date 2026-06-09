import asyncio
from typing import Generator

import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine import CheckpointEngineRegistry, CheckpointEngineWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.workers.config import CheckpointEngineConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout import BaseRollout, RolloutReplica
from verl.single_controller.base import Worker
import os
import time
from verl.utils.device import get_torch_device
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity

def generate_random_weights(size = 2048 * 2048):
    torch.random.manual_seed(192767)
    weights = []
    for i in range(32):
        weights.append(
            (f"weight_{i}", torch.arange(start=i*size, end=(i+1)*size, dtype = torch.int64, device="cuda"))
        )
    return weights


class MockTrainingWorker(Worker):

    def __init__(self, checkpoint_engine_config: CheckpointEngineConfig) -> None:
        Worker.__init__(self)


        # dist stuff
        initialize_global_process_group_ray(timeout_second=None)
        set_numa_affinity()
        
        # CE init
        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        if torch.distributed.get_rank() == 0:
            engine_kwargs["is_master"] = True
        self.checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=bucket_size, **engine_kwargs)

        self.weights = generate_random_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        per_tensor_param = self.weights
        await self.checkpoint_engine.send_weights(per_tensor_param)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

class MockServerAdapter(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, check_allclose: bool = True):
        super().__init__(config, model_config, device_mesh=None)
        self.check_allclose = check_allclose
        self.weights = generate_random_weights()
        self.received_weights: dict[str, torch.Tensor] = {}

    async def resume(self, tags: list[str]):
        raise NotImplementedError()

    async def release(self):
        raise NotImplementedError()

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        async for name, weight in weights:
            if self.check_allclose:
                self.received_weights[name] = torch.empty_like(weight, device="cuda")
                # print(f"copying..., weight is on={weight.device}")
                self.received_weights[name].copy_(weight, non_blocking=True)

    def check_weights(self):
        if not self.check_allclose:
            return
        
        for name, weight in self.weights:
            assert name in self.received_weights, f"weight {name} not received"
            received = self.received_weights[name].flatten()
            target = weight.to(received.device).flatten()

            mask = ~(target == received)
            idx = mask.nonzero()
            if (mask.any()):
                print(f"{name} mismatch at {idx.shape[0]}")
            # print(f"{name}:\n{weight.to(received.device)}\n{received}")
            torch.testing.assert_close(target, received)
            # assert torch.allclose(weight.to(received.device), received), f"weight {name} not equal: {weight.to(received.device)} vs {received}"
        self.received_weights.clear()
        print("Check passed, all weights are equal!")


class MockReplica(RolloutReplica):
    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        raise NotImplementedError

    async def launch_servers(self):
        """Launch http server in each node."""
        raise NotImplementedError


class MockCheckpointEngineWorker(CheckpointEngineWorker):
    def __init__(
        self, rollout_config: RolloutConfig, model_config: HFModelConfig, check_allclose: bool = True, *args, **kwargs
    ) -> None:
        server_adapter = MockServerAdapter(rollout_config, model_config, check_allclose)
        super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_weights(self):
        self.server_adapter.check_weights()

def create_mock_trainer_wg(
    resource_pool: RayResourcePool, checkpoint_engine_config: CheckpointEngineConfig
) -> RayWorkerGroup:
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(MockTrainingWorker),
        checkpoint_engine_config=checkpoint_engine_config,
    )
    ray_cls_with_init.update_options(
        {
            "runtime_env": {
                "env_vars": {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            }
        }
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())
    return wg

async def create_mock_rollout_wg(
    resource_pool: RayResourcePool,
    model_config: HFModelConfig,
    rollout_config: RolloutConfig,
    check_allclose: bool = True,
) -> tuple[RayWorkerGroup, list[MockReplica]]:
    # create rollout worker group
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(MockCheckpointEngineWorker),
        model_config=model_config,
        rollout_config=rollout_config,
        check_allclose=check_allclose,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())

    # create rollout replicas
    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    num_replicas = wg.world_size // rollout_world_size
    replicas = []
    for replica_rank in range(num_replicas):
        replica = MockReplica(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
        )
        replicas.append(replica)
    await asyncio.gather(*[replica.init_hybrid(wg) for replica in replicas])

    return wg, replicas

