from verl.experimental.reward_loop import RewardLoopManager
import ray
from verl import DataProto
import hydra
import pyarrow.parquet as pq

@hydra.main(
    config_path="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/verl/verl/experimental/fully_async_policy/config", 
    config_name="fully_async_ppo_trainer", 
    version_base=None
)
def validate_data_main(config):
    ray.init(num_cpus=8)

    manager = RewardLoopManager(config)
    train_data = config.data.train_file

    train_ds = pq.ParquetFile(train_data)
    train_ds.read_row_group
    for batch in train_ds.iter_batches():
        for row in batch.to_pylist():
            result = manager.compute_rm_score(
                data = DataProto.from_dict(
                    tensor = None,
                    non_tensor = row
                )
            )
            print(result)

    ray.stop()