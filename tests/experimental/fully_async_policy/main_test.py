import hydra
from mock_utils import MockFullyAsyncTaskRunner
import ray

@hydra.main(
    config_path="./config", 
    config_name="sglang", 
    version_base=None
)
def mock_run_ppo(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(MockFullyAsyncTaskRunner))
    print(f"total time: {time() - start_time:.2f} seconds")



if __name__ == "__main__":
    ray.init(
        labels={"actor": "true", "rollout": "true", "trainer_head": "true", "rollout_head": "true"},
        num_cpus=288
    )
    mock_run_ppo()
    ray.shutdown()