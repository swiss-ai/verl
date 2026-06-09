import pytest
import ray
from tests.checkpoint_engine.mock_utils import create_mock_trainer_wg, create_mock_rollout_wg
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray.base import (
    RayResourcePool,
    split_resource_pool,
)
from verl.utils.device import get_device_name
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
import torch

@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("rebuild_group", [False])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 2)])
async def test_mooncake_checkpoint_engine(
    rebuild_group,
    num_trainer,
    num_rollout,
    device,
    num_nodes=1,
    num_gpus_per_node=4,
    check_allclose=True,
    model_path="swiss-ai/Apertus-8B-Instruct-2509",
):
    # model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

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
    for _ in range(8):
        await checkpoint_manager.update_weights()
        rollout.check_weights()
    ray.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_mooncake_checkpoint_engine(
        rebuild_group=False,
        num_trainer=2,
        num_rollout=2,
        num_gpus_per_node=4,
        device="cuda"
    ))