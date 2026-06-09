import ray
import torch

from tests.checkpoint_engine.mock_utils import create_mock_trainer_wg, create_mock_rollout_wg
from tests.checkpoint_engine.test_utils import create_rollout_worker_group, create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray.base import (
    RayResourcePool,
    split_resource_pool,
)
from verl.utils.device import get_device_name
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig

async def multi_node_cxi_test(
    node_rank,
    rebuild_group,
    num_trainer,
    num_rollout,
    device,
    num_nodes,
    num_gpus_per_node=4,
    check_allclose=True,
    num_iter = 8,
    model_path="swiss-ai/Apertus-8B-Instruct-2509",
):
    _ray_runtime_env = {
        "env_vars": {
            # "ASCEND_USE_SHORT_CONNECTION": "1",
            "VERL_LOGGING_LEVEL": "DEBUG",
        }
    }
    if (node_rank == 0):
        ray.init(
            runtime_env=_ray_runtime_env
        )
    else:
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        ray.init(
            runtime_env=_ray_runtime_env,
            address=f"{master_addr}:{master_port}"
        )
        while (ray.is_initialized()):
            await asyncio.sleep(20)
        return
    

    # initialize config
    backend = "cxi_sharded"
    checkpoint_engine_config_trainer = CheckpointEngineConfig(
        backend=backend, update_weights_bucket_megabytes=2048, engine_kwargs={backend: {"device": device, "rebuild_group": rebuild_group, "is_trainer": True}}
    )
    checkpoint_engine_config_rollout = CheckpointEngineConfig(
        backend=backend, update_weights_bucket_megabytes=2048, engine_kwargs={backend: {"device": device, "rebuild_group": rebuild_group, "is_trainer": False}}
    )

    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="sglang", checkpoint_engine=checkpoint_engine_config_rollout)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=1)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config_trainer)
    trainer.reset()
    rollout, replicas = await create_rollout_worker_group(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(config=checkpoint_engine_config_trainer, trainer=trainer, replicas=replicas)
    for _ in range(num_iter):
        await checkpoint_manager.update_weights()
        rollout.check_weights()
    ray.shutdown()

async def mock_multi_node_cxi_test(
    node_rank,
    rebuild_group,
    num_trainer,
    num_rollout,
    device,
    num_nodes,
    num_gpus_per_node=4,
    check_allclose=False,
    num_iter = 8,
    model_path="swiss-ai/Apertus-8B-Instruct-2509",
):
    _ray_runtime_env = {
        "env_vars": {
            # "ASCEND_USE_SHORT_CONNECTION": "1",
            "VERL_LOGGING_LEVEL": "DEBUG",
        }
    }
    if (node_rank == 0):
        ray.init(
            runtime_env=_ray_runtime_env
        )
    else:
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        ray.init(
            runtime_env=_ray_runtime_env,
            address=f"{master_addr}:{master_port}"
        )
        while (ray.is_initialized()):
            await asyncio.sleep(20)
        return
    

    # initialize config
    backend = "cxi_sharded"
    checkpoint_engine_config_trainer = CheckpointEngineConfig(
        backend=backend, update_weights_bucket_megabytes=512, engine_kwargs={backend: {"device": device, "rebuild_group": rebuild_group, "rollout_dtype": torch.int64, "is_trainer": True}}
    )
    checkpoint_engine_config_rollout = CheckpointEngineConfig(
        backend=backend, update_weights_bucket_megabytes=512, engine_kwargs={backend: {"device": device, "rebuild_group": rebuild_group, "rollout_dtype": torch.int64, "is_trainer": False}}
    )

    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="sglang", checkpoint_engine=checkpoint_engine_config_rollout)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=1)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    trainer = create_mock_trainer_wg(trainer_pool, checkpoint_engine_config_trainer)
    # trainer.reset() doesn't do anything
    rollout, replicas = await create_mock_rollout_wg(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(config=checkpoint_engine_config_trainer, trainer=trainer, replicas=replicas)
    for _ in range(num_iter):
        await checkpoint_manager.update_weights()
        rollout.check_weights()
    ray.shutdown()


if __name__ == "__main__":
    import os
    import asyncio
    node_id = int(os.environ["SLURM_NODEID"])
    nnodes = int(os.environ["NNODES"])

    gpus_per_node = 4
    num_trainer = (nnodes // 2) * gpus_per_node
    num_rollout = (nnodes // 2) * gpus_per_node
    asyncio.run(
        multi_node_cxi_test(
            node_rank=node_id,
            rebuild_group=False,
            num_nodes=nnodes,
            num_trainer=num_trainer,
            num_rollout=num_rollout,
            num_gpus_per_node = gpus_per_node,
            device = "cuda"
        )
    )
