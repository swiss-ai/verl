# WORK IN PROGRESS: not ready

from verl.experimental.agent_loop import AgentLoopManager
from verl.workers.rollout.llm_server import LLMServerClient, TokenOutput
import ray
import hydra
from verl.protocol import DataProto
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

class MockLLMClient(LLMServerClient):

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data = None, video_data = None, audio_data = None, mm_processor_kwargs = None, **kwargs):
        return TokenOutput(token_ids=prompt_ids)


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
    config_path=".", 
    config_name="async_single_node", 
    version_base=None
)
def validate_data_main(config):
    ray.init(num_cpus=1)

    mock_client = MockLLMClient(config=config)
    agent_loop_manager = AgentLoopManager.create(
        config = config,
        llm_client = mock_client
    )
    n = 2
    config.actor_rollout_ref.rollout.n = n
    raw_prompts = [
        [
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "user", "content": "What's the temperature in Los Angeles now?"},
        ],
        [
            {"role": "user", "content": "What's the temperature in New York now?"},
        ],
        [
            {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
        ],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(prompt) for prompt in raw_prompts], dtype=object),
            "agent_name": np.array(["tool_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
            "extra_info": np.array([{'tool_selection': ['display_answers']}] * len(raw_prompts))
        },
    )
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    ray.shutdown()

if __name__ == "__main__":
    validate_data_main()