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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.output_format import output_format_enabled


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [
        k
        for k in non_tensor_batch_keys_to_pop
        if k in full_batch.non_tensor_batch.keys()
    ]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Setting selected agent, that supports partial
    if not config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(
            ["single_turn_agent"] * len(full_batch), dtype=object
        )

    # Add global step count to generated data
    full_batch = full_batch.repeat(
        repeat_times=config.actor_rollout_ref.rollout.n, interleave=True
    )
    return full_batch


def validate_inverse_batch(rollout_config) -> None:
    """Validate fully async inverse-batch rollout settings."""
    rollout_n = int(rollout_config.n)
    n_per_round = int(rollout_config.n_per_round)

    if rollout_n < 1:
        raise ValueError(f"actor_rollout_ref.rollout.n must be >= 1, got {rollout_n}")
    if n_per_round < 1:
        raise ValueError(f"actor_rollout_ref.rollout.n_per_round must be >= 1, got {n_per_round}")
    if n_per_round > rollout_n:
        raise ValueError(
            "actor_rollout_ref.rollout.n_per_round must be <= actor_rollout_ref.rollout.n, "
            f"got n_per_round={n_per_round}, n={rollout_n}"
        )
    if rollout_n % n_per_round != 0:
        raise ValueError(
            "actor_rollout_ref.rollout.n_per_round must divide actor_rollout_ref.rollout.n, "
            f"got n_per_round={n_per_round}, n={rollout_n}"
        )


def adaptive_group_size_enabled(config) -> bool:
    adaptive_config = config.actor_rollout_ref.rollout.get("adaptive_group_size", {})
    return bool(adaptive_config.get("enabled", False))


def speculative_prompt_concurrency_enabled(config) -> bool:
    adaptive_config = config.actor_rollout_ref.rollout.get("adaptive_group_size", {})
    speculative_config = adaptive_config.get("speculative_prompt_concurrency", {})
    return bool(speculative_config.get("enabled", False))


def validate_adaptive_group_size_config(config) -> None:
    """Validate the deliberately narrow first fully-async adaptive implementation."""
    adaptive_config = config.actor_rollout_ref.rollout.get("adaptive_group_size", {})
    speculative_config = adaptive_config.get("speculative_prompt_concurrency", {})
    speculative_enabled = bool(speculative_config.get("enabled", False))

    if speculative_enabled and not adaptive_group_size_enabled(config):
        raise ValueError("speculative_prompt_concurrency requires adaptive_group_size.enabled=True")
    if not adaptive_group_size_enabled(config):
        return

    max_num_rounds = int(adaptive_config.max_num_rounds)
    if max_num_rounds < 1:
        raise ValueError(
            "actor_rollout_ref.rollout.adaptive_group_size.max_num_rounds must be >= 1, "
            f"got {max_num_rounds}"
        )

    filter_config = config.algorithm.get("filter_groups", None)
    if filter_config is None or not filter_config.enable:
        raise ValueError("adaptive_group_size requires algorithm.filter_groups.enable=True")
    if filter_config.metric not in {"acc", "task_success"}:
        raise ValueError(
            "adaptive_group_size supports filter_groups.metric in {'acc', 'task_success'}, "
            f"got {filter_config.metric!r}"
        )

    if str(config.algorithm.adv_estimator) != "rloo_vectorized":
        raise ValueError("adaptive_group_size currently requires algorithm.adv_estimator=rloo_vectorized")
    rollout_correction = config.algorithm.get("rollout_correction", None)
    if rollout_correction is None or not rollout_correction.get("bypass_mode", False):
        raise ValueError("adaptive_group_size currently requires algorithm.rollout_correction.bypass_mode=True")
    if not config.async_training.get("use_rollout_log_probs", False):
        raise ValueError("adaptive_group_size currently requires async_training.use_rollout_log_probs=True")
    if config.actor_rollout_ref.actor.get("use_prefix_grouper", False):
        raise ValueError("adaptive_group_size currently does not support actor.use_prefix_grouper=True")

    from verl.trainer.distillation.losses import is_distillation_enabled
    from verl.trainer.ppo.utils import need_critic, need_reference_policy

    if need_critic(config):
        raise ValueError("adaptive_group_size currently does not support critic training")
    if need_reference_policy(config):
        raise ValueError("adaptive_group_size currently does not support a reference policy")
    if is_distillation_enabled(config.get("distillation")):
        raise ValueError("adaptive_group_size currently does not support distillation")

    if speculative_enabled:
        target = int(speculative_config.get("target_inflight_trajectories_per_replica", 40))
        if target < 1:
            raise ValueError(
                "adaptive_group_size.speculative_prompt_concurrency."
                f"target_inflight_trajectories_per_replica must be >= 1, got {target}"
            )
        max_num_seqs = int(config.actor_rollout_ref.rollout.max_num_seqs)
        if target > max_num_seqs:
            raise ValueError(
                "adaptive_group_size.speculative_prompt_concurrency."
                "target_inflight_trajectories_per_replica must be <= rollout.max_num_seqs, "
                f"got target={target}, max_num_seqs={max_num_seqs}"
            )


def validate_async_filter_groups_config(config, logger=None):
    """Validate filter_groups settings in config."""
    filter_groups_config = config.algorithm.get("filter_groups", None)
    if filter_groups_config is None or not filter_groups_config.enable:
        return

    metric_name = filter_groups_config.metric
    if not metric_name:
        raise ValueError(
            "algorithm.filter_groups.metric must be set when filter_groups.enable=True"
        )

    if output_format_enabled(config) and metric_name != "task_success":
        raise ValueError(
            "output_format.enabled=true with group filtering requires "
            "algorithm.filter_groups.metric=task_success"
        )

    if metric_name == "seq_final_reward" and config.algorithm.use_kl_in_reward:
        raise ValueError(
            "algorithm.filter_groups.metric='seq_final_reward' is not supported in fully async mode when "
            "algorithm.use_kl_in_reward=True because KL-adjusted rewards are computed on the trainer."
        )

    min_group = filter_groups_config.get("min", None)
    max_group = filter_groups_config.get("max", None)
    if min_group is None or max_group is None:
        raise ValueError(
            "algorithm.filter_groups.min and algorithm.filter_groups.max must be set"
        )
    if min_group >= max_group:
        raise ValueError(
            f"algorithm.filter_groups.min must be less than algorithm.filter_groups.max, got {min_group} >= {max_group}"
        )

    max_num_gen_batches = filter_groups_config.get("max_num_gen_batches", 0)
    if max_num_gen_batches > 0 and logger is not None:
        logger.warning(
            "algorithm.filter_groups.max_num_gen_batches=%s is ignored in fully async streaming mode; "
            "filtered samples are simply skipped before queue insertion.",
            max_num_gen_batches,
        )


def get_async_filter_metric_values(batch: DataProto, filter_groups_config) -> np.ndarray:
    """Extract one scalar filtering metric per trajectory."""
    metric_name = filter_groups_config.metric
    if metric_name in {"seq_reward", "seq_final_reward"}:
        if batch.batch is None or "rm_scores" not in batch.batch.keys():
            raise ValueError(
                f"algorithm.filter_groups.metric={metric_name!r} requires 'rm_scores' in the rollout output"
            )
        metric_values = batch.batch["rm_scores"].sum(dim=-1)
    else:
        if metric_name not in batch.non_tensor_batch:
            raise ValueError(
                f"algorithm.filter_groups.metric={metric_name!r} was not found in rollout reward extras. "
                "Use a metric returned by the reward function or one of: seq_reward, seq_final_reward."
            )
        metric_values = batch.non_tensor_batch[metric_name]

    if isinstance(metric_values, torch.Tensor):
        metric_values = metric_values.detach().cpu().numpy()
    metric_values = np.asarray(metric_values, dtype=np.float64).reshape(-1)
    if len(metric_values) != len(batch):
        raise ValueError(
            f"algorithm.filter_groups.metric={metric_name!r} produced {len(metric_values)} values "
            f"for {len(batch)} trajectories"
        )
    return metric_values


def adaptive_group_has_success(batch: DataProto, filter_groups_config) -> bool:
    """Return whether a completed logical sampling round contains a fully correct trajectory."""
    metric_values = get_async_filter_metric_values(batch, filter_groups_config)
    correct_threshold = 0.8
    return bool(np.any(np.isfinite(metric_values) & (metric_values >= correct_threshold)))


def should_keep_async_filter_group(batch: DataProto, filter_groups_config) -> bool:
    """Return whether a fully async rollout group should be enqueued for training."""
    metric_values = get_async_filter_metric_values(batch, filter_groups_config)
    return bool(
        np.std(metric_values) > 0
        and filter_groups_config.min < np.mean(metric_values) < filter_groups_config.max
    )


def should_keep_output_format_group(batch: DataProto) -> bool:
    """Filter using binary task success and output-format contrast."""
    required_keys = ("task_success", "format_valid")
    missing_keys = [key for key in required_keys if key not in batch.non_tensor_batch]
    if missing_keys:
        raise ValueError(
            "output-format group filtering requires reward extras: "
            + ", ".join(missing_keys)
        )

    task_success = np.asarray(batch.non_tensor_batch["task_success"], dtype=np.float64).reshape(-1)
    format_valid = np.asarray(batch.non_tensor_batch["format_valid"], dtype=bool).reshape(-1)
    if len(task_success) != len(batch) or len(format_valid) != len(batch):
        raise ValueError("output-format filter metrics must contain one value per trajectory")
    if not np.all(np.isclose(task_success, 0.0) | np.isclose(task_success, 1.0)):
        raise ValueError("output-format task_success values must be binary")

    successful = np.isclose(task_success, 1.0)
    if not np.any(successful):
        return False
    if not np.all(successful):
        return True
    return bool(np.any(format_valid) and not np.all(format_valid))


def addition_process(output: DataProto):
    """collect metirics"""
    metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
    if isinstance(metrics, dict):
        processing_times_list = metrics["generate_sequences"]
        tool_calls_times_list = metrics["tool_calls"]
    else:
        processing_times_list = [item["generate_sequences"] for item in metrics]
        tool_calls_times_list = [item["tool_calls"] for item in metrics]
    output.non_tensor_batch["processing_times"] = processing_times_list
    output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
    return output


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(
        f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects"
    )

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Preserve user-facing rollout metrics; namespace internal rollouter state.
    rollout_status = {
        key if key.startswith("rollout/") else f"fully_async/{key}": value
        for key, value in rollout_status.items()
    }

    for rs in rollout_samples:
        batch = addition_process(rs.full_batch)
        rollout_samples_batch.append(batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(
            final_batch.batch["attention_mask"], dim=-1
        ).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {
        f"fully_async/{key}": value for key, value in processing_time_stats.items()
    }

    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [
        abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)
    ]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0)
        / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


def partition_adaptive_prompt_minibatches(
    batch: DataProto,
    *,
    prompts_per_minibatch: int,
    num_minibatches: int,
    seed: int,
) -> list[DataProto]:
    """Partition a variable-trajectory batch by prompt UID, never splitting a group."""
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("adaptive prompt minibatching requires a 'uid' field")

    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object).reshape(-1)
    unique_uids = list(dict.fromkeys(uids.tolist()))
    expected_prompts = prompts_per_minibatch * num_minibatches
    if len(unique_uids) != expected_prompts:
        raise ValueError(
            "adaptive prompt minibatching requires exactly "
            f"{expected_prompts} prompt UIDs, got {len(unique_uids)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_uids)
    minibatches = []
    for start in range(0, expected_prompts, prompts_per_minibatch):
        selected_uids = set(unique_uids[start : start + prompts_per_minibatch])
        selected_indices = np.flatnonzero(np.fromiter((uid in selected_uids for uid in uids), dtype=bool))
        minibatch = batch.select_idxs(selected_indices)
        minibatch.meta_info = dict(batch.meta_info)
        minibatches.append(minibatch)
    return minibatches


def pad_adaptive_minibatch(batch: DataProto, *, batch_multiple: int) -> tuple[DataProto, int]:
    """Append duplicated zero-loss rows until a prompt minibatch is DP-divisible."""
    if batch_multiple < 1:
        raise ValueError(f"batch_multiple must be >= 1, got {batch_multiple}")
    padding_size = (-len(batch)) % batch_multiple

    batch.non_tensor_batch["is_padding"] = np.zeros(len(batch), dtype=bool)
    if padding_size == 0:
        return batch, 0

    padding = batch.select_idxs([0]).repeat(padding_size)
    zero_tensor_keys = {
        "response_mask",
        "loss_mask",
        "advantages",
        "returns",
        "rewards",
        "token_level_rewards",
        "token_level_scores",
        "rm_scores",
        "rollout_log_probs",
        "old_log_probs",
        "importance_weights",
        "importance_sampling_ratio",
        "rollout_is_weights",
    }
    if padding.batch is not None:
        for key in zero_tensor_keys.intersection(padding.batch.keys()):
            padding.batch[key] = torch.zeros_like(padding.batch[key])

    padding_uid = "__adaptive_padding__"
    padding.non_tensor_batch["uid"] = np.full(padding_size, padding_uid, dtype=object)
    padding.non_tensor_batch["is_padding"] = np.ones(padding_size, dtype=bool)
    # ``select_idxs`` and ``repeat`` retain the source metadata.  The padding
    # rows do not own metadata, and passing the inherited copy to concat makes
    # DataProto compare values such as trajectory_param_versions with scalar
    # equality.  Array-valued metadata then raises an ambiguous-truth error.
    # Keep the real minibatch metadata as the single source of truth instead.
    padding.meta_info = {}
    padded = DataProto.concat([batch, padding])
    padded.meta_info = dict(batch.meta_info)
    return padded, padding_size


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "fully_async/count/filtered_group_samples",
                "fully_async/count/filtered_group_trajectories",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(
        self, metrics: dict[str, Any], sample_count: int, timestamp: float = None
    ):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(
                v * c for v, c in zip(values, self.sample_counts, strict=False)
            )
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = (
                aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]
            )

        # perf/throughput
        REQUIRED_PERF_KEYS = {
            "perf/throughput",
            "perf/total_num_tokens",
            "perf/time_per_step",
        }
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = (
                aggregated["timing_s/gen"] / aggregated["timing_s/step"]
            )

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }


def task_exception_handler(task: asyncio.Task):
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
    return task
