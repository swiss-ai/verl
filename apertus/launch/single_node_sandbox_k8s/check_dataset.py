from verl.experimental.reward_loop import RewardLoopManager
import ray
from verl import DataProto
import hydra
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds

def reward_loop_reward_fn(data: DataProto, reward_loop: RewardLoopManager):
    """
    Replicate the logic inside RewardLoopManager, avoid materializing prompts and only get the outputs
    """
    chunks = data.chunk(len(reward_loop.reward_loop_workers))
    outputs = ray.get(
        [
            worker.compute_score_batch.remote(chunk)
            for worker, chunk in zip(reward_loop.reward_loop_workers, chunks, strict=True)
        ]
    )
    outputs_flat = [item for sublist in outputs for item in sublist]
    return outputs_flat

def get_uniques(
    files,
    group_col,
    n_samples = 4
):
    group_batches = {}
    group_counts = {}
    sources = set()

    for path in files:
        train_ds = pq.ParquetFile(path)
        for batch in train_ds.iter_batches(batch_size=8192):
            unique_sources = pc.unique(batch.column(group_col)).to_pylist()
            for source_val in unique_sources:
                if source_val in sources and group_counts[source_val] >= n_samples:
                    continue
                mask = pc.equal(batch.column(group_col), source_val)
                filtered_batch = batch.filter(mask)

                gc = 0 if source_val not in sources else group_counts[source_val]
                needed = n_samples - gc
                sliced_batch = filtered_batch.slice(0, needed)
                if (source_val in sources):
                    group_batches[source_val].append(sliced_batch)
                    group_counts[source_val] += len(sliced_batch)
                else:
                    group_batches[source_val] = [sliced_batch]
                    group_counts[source_val] = len(sliced_batch)
                    sources.add(source_val)

    result = [b for batches in group_batches.values() for b in batches]
    return result



@hydra.main(
    config_path="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/verl/verl/experimental/fully_async_policy/config", 
    config_name="fully_async_ppo_trainer", 
    version_base=None
)
def validate_data_main(config):
    ray.init(num_cpus=1)

    config.reward.num_workers = 1

    manager = RewardLoopManager(config)
    train_files = config.data.train_files
    for batch in get_uniques(files = train_files, group_col = "data_source"):
        data = batch.to_pydict()
        data = DataProto.from_dict(
            tensors = None,
            non_tensors = batch.to_pydict()
        )
        extra_info = data.non_tensor_batch.get("extra_info")
        for obj in extra_info:
            obj["response_text"] = "test"
        result = reward_loop_reward_fn(data, manager)
        print(result)
    ray.shutdown()

if __name__ == "__main__":
    validate_data_main()