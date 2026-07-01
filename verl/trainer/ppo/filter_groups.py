from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from verl import DataProto


def validate_filter_groups_config(config) -> None:
    """Validate algorithm.filter_groups settings shared by sync filtering."""
    filter_groups_config = config.algorithm.get("filter_groups", None)
    if filter_groups_config is None or not filter_groups_config.enable:
        return

    metric_name = filter_groups_config.metric
    if not metric_name:
        raise ValueError("algorithm.filter_groups.metric must be set when filter_groups.enable=True")

    if metric_name == "seq_final_reward" and config.algorithm.use_kl_in_reward:
        raise ValueError(
            "algorithm.filter_groups.metric='seq_final_reward' is not supported when "
            "algorithm.use_kl_in_reward=True because KL-adjusted rewards are computed later in the trainer."
        )

    min_group = filter_groups_config.get("min", None)
    max_group = filter_groups_config.get("max", None)
    if min_group is None or max_group is None:
        raise ValueError("algorithm.filter_groups.min and algorithm.filter_groups.max must be set")
    if min_group >= max_group:
        raise ValueError(
            f"algorithm.filter_groups.min must be less than algorithm.filter_groups.max, got {min_group} >= {max_group}"
        )


def get_filter_group_metric_values(batch: DataProto, metric_name: str) -> np.ndarray:
    """Return one scalar filtering metric per trajectory."""
    if metric_name in {"seq_reward", "seq_final_reward"}:
        if batch.batch is None or "rm_scores" not in batch.batch.keys():
            raise ValueError(f"algorithm.filter_groups.metric={metric_name!r} requires 'rm_scores' in the batch")
        metric_values = batch.batch["rm_scores"].sum(dim=-1)
    else:
        if metric_name not in batch.non_tensor_batch:
            raise ValueError(
                f"algorithm.filter_groups.metric={metric_name!r} was not found in reward extras. "
                "Use a metric returned by the reward function or one of: seq_reward, seq_final_reward."
            )
        metric_values = batch.non_tensor_batch[metric_name]

    if isinstance(metric_values, torch.Tensor):
        metric_values = metric_values.detach().cpu().numpy()
    return np.asarray(metric_values, dtype=np.float64).reshape(-1)


def filter_groups(batch: DataProto, filter_groups_config) -> tuple[DataProto | None, dict[str, float]]:
    """Keep only prompt groups whose configured reward metric passes the group filter (threshold and non-zero std)."""
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("algorithm.filter_groups requires 'uid' in batch.non_tensor_batch")

    metric_name = filter_groups_config.metric
    metric_values = get_filter_group_metric_values(batch, metric_name)
    uids = np.asarray(batch.non_tensor_batch["uid"])
    if len(metric_values) != len(batch):
        raise ValueError(
            f"algorithm.filter_groups.metric={metric_name!r} produced {len(metric_values)} values for {len(batch)} rows"
        )

    group_indices: OrderedDict[object, list[int]] = OrderedDict()
    for idx, uid in enumerate(uids):
        group_indices.setdefault(uid, []).append(idx)

    keep_mask = np.zeros(len(batch), dtype=bool)
    kept_metric_sum = 0.0
    kept_metric_count = 0
    all_correct_groups = 0
    all_incorrect_groups = 0
    below_min_groups = 0
    above_max_groups = 0

    for indices in group_indices.values():
        group_values = metric_values[indices]
        group_mean = float(np.mean(group_values))
        all_correct_groups += int(np.all(np.isclose(group_values, 1.0)))
        all_incorrect_groups += int(np.all(np.isclose(group_values, 0.0)))
        below_min_groups += int(group_mean <= filter_groups_config.min)
        above_max_groups += int(group_mean >= filter_groups_config.max)
        keep_group = bool(
            np.std(group_values) > 0    # Always enforce non-zero std to avoid zero-advantage groups
            and filter_groups_config.min < group_mean < filter_groups_config.max    # Group-accuracy filter
        )
        if keep_group:
            keep_mask[indices] = True
            kept_metric_sum += float(np.sum(group_values))
            kept_metric_count += len(indices)

    filtered_batch = batch.select_idxs(keep_mask) if np.any(keep_mask) else None
    total_groups = len(group_indices)
    kept_groups = len({uids[idx] for idx in np.flatnonzero(keep_mask)})

    metrics = {
        "filter_groups/count/generated_prompts": total_groups,
        "filter_groups/count/kept_prompts": kept_groups,
        "filter_groups/count/filtered_prompts": total_groups - kept_groups,
        "filter_groups/count/all_correct_prompts": all_correct_groups,
        "filter_groups/count/all_incorrect_prompts": all_incorrect_groups,
        "filter_groups/count/below_min_prompts": below_min_groups,
        "filter_groups/count/above_max_prompts": above_max_groups,
        "filter_groups/count/outside_interval_prompts": below_min_groups + above_max_groups,
        f"filter_groups/pre_filter/{metric_name}/mean": float(np.mean(metric_values))
        if len(metric_values) > 0
        else float("nan"),
        f"filter_groups/post_filter/{metric_name}/mean": kept_metric_sum / kept_metric_count
        if kept_metric_count > 0
        else float("nan"),
    }
    return filtered_batch, metrics


def select_first_groups(batch: DataProto, max_groups: int) -> tuple[DataProto | None, int]:
    """Select the leading complete uid groups."""
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("select_first_groups requires 'uid' in batch.non_tensor_batch")

    uids = np.asarray(batch.non_tensor_batch["uid"])
    group_indices: OrderedDict[object, list[int]] = OrderedDict()
    for idx, uid in enumerate(uids):
        group_indices.setdefault(uid, []).append(idx)

    selected_indices = []
    for indices in list(group_indices.values())[:max_groups]:
        selected_indices.extend(indices)

    if not selected_indices:
        return None, 0
    return batch.select_idxs(np.asarray(selected_indices, dtype=np.int64)), len(selected_indices)
