"""Utilities for adaptive grouped RL sampling in PPO trainers.

This module only contains stateless helper functions used by adaptive
group sampling. It intentionally avoids trainer state and side effects.
"""

from typing import Any

import numpy as np


def select_balanced_rollout_entries(
    positive_entries: list[dict[str, Any]],
    negative_entries: list[dict[str, Any]],
    target_count: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Select a fixed-size, as-balanced-as-possible subset from class caches."""
    target_pos = target_count // 2
    target_neg = target_count - target_pos

    pos_take = min(target_pos, len(positive_entries))
    neg_take = min(target_neg, len(negative_entries))

    selected = positive_entries[:pos_take] + negative_entries[:neg_take]

    missing = target_count - len(selected)
    if missing > 0:
        remaining_pos = positive_entries[pos_take:]
        remaining_neg = negative_entries[neg_take:]
        # Prefer filling from the class with more remaining candidates.
        if len(remaining_pos) >= len(remaining_neg):
            fill_pool = remaining_pos + remaining_neg
        else:
            fill_pool = remaining_neg + remaining_pos
        selected.extend(fill_pool[:missing])

    selected = selected[:target_count]
    selected_pos = sum(1 for entry in selected if entry["is_positive"])
    selected_neg = len(selected) - selected_pos
    return selected, selected_pos, selected_neg


def extract_reward_extra_info_row(reward_extra_infos_dict: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    """Extract one sample's extra reward info from batched extras."""
    row = {}
    for key, values in reward_extra_infos_dict.items():
        if isinstance(values, np.ndarray):
            if values.ndim > 0 and values.shape[0] == batch_size:
                row[key] = values[index]
        elif isinstance(values, list):
            if len(values) == batch_size:
                row[key] = values[index]
        else:
            row[key] = values
    return row


def prune_prompt_caches(prompt_state: dict[str, Any], target_rollouts: int) -> None:
    """Keep only the minimum cache needed to still form a balanced final group."""
    target_pos = target_rollouts // 2
    target_neg = target_rollouts - target_pos

    max_pos_to_keep = max(target_pos, target_rollouts - prompt_state["neg"])
    max_neg_to_keep = max(target_neg, target_rollouts - prompt_state["pos"])

    if len(prompt_state["positive_cache"]) > max_pos_to_keep:
        prompt_state["positive_cache"] = prompt_state["positive_cache"][:max_pos_to_keep]
    if len(prompt_state["negative_cache"]) > max_neg_to_keep:
        prompt_state["negative_cache"] = prompt_state["negative_cache"][:max_neg_to_keep]


def finalize_prompt_rollouts(prompt_state: dict[str, Any], target_rollouts: int) -> None:
    """Finalize one prompt cache into exactly target_rollouts selected samples."""
    selected, selected_pos, selected_neg = select_balanced_rollout_entries(
        positive_entries=prompt_state["positive_cache"],
        negative_entries=prompt_state["negative_cache"],
        target_count=target_rollouts,
    )
    if len(selected) < target_rollouts:
        raise ValueError(f"Unable to select {target_rollouts} rollouts from prompt cache; only got {len(selected)}.")

    prompt_state["selected"] = selected
    prompt_state["selected_pos"] = selected_pos
    prompt_state["selected_neg"] = selected_neg
    prompt_state["finalized"] = True
