"""Utilities for adaptive grouped RL sampling in PPO trainers.

This module only contains stateless helper functions used by adaptive
group sampling. It intentionally avoids trainer state and side effects.
"""

from typing import Any

import numpy as np
import torch

from verl import DataProto

ADAPTIVE_KEEP_ALL_PAD_MASK_KEY = "_adaptive_keep_all_is_pad"


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


def finalize_prompt_rollouts_keep_all(prompt_state: dict[str, Any]) -> None:
    """Finalize one prompt cache by keeping all cached samples.

    This is used when adaptive_group_sampling.apply_downsampling=False.
    In that mode we intentionally keep every cached rollout so the actor/critic update
    can consume a variable number of samples per prompt.
    """
    # Preserve deterministic ordering: positives first then negatives, matching cache append order.
    selected = prompt_state["positive_cache"] + prompt_state["negative_cache"]

    if not selected:
        raise ValueError("Unable to finalize keep-all rollouts: prompt cache is empty.")

    selected_pos = sum(1 for entry in selected if entry["is_positive"])
    selected_neg = len(selected) - selected_pos

    prompt_state["selected"] = selected
    prompt_state["selected_pos"] = selected_pos
    prompt_state["selected_neg"] = selected_neg
    prompt_state["finalized"] = True


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


def _build_padding_indices(batch_size: int, pad_size: int) -> list[int]:
    """Build deterministic indices for DataProto padding by repeating from batch head."""
    if pad_size <= 0:
        return list(range(batch_size))
    if batch_size <= 0:
        raise ValueError("Cannot pad an empty batch.")
    return list(range(batch_size)) + [idx % batch_size for idx in range(pad_size)]


def _apply_padding_mask_to_response_mask(batch: DataProto, mask_key: str) -> None:
    """Force padded samples to have zero response mask so they do not contribute to loss.

    Args:
        batch: Target batch that may contain adaptive padding rows.
        mask_key: non-tensor key holding a boolean array aligned with dim-0.
            Entries with ``True`` are treated as adaptive padding rows.
    """
    if "response_mask" not in batch.batch.keys():
        return
    if mask_key not in batch.non_tensor_batch:
        return

    pad_mask = np.asarray(batch.non_tensor_batch[mask_key]).astype(bool)
    if pad_mask.shape[0] != len(batch):
        raise ValueError(
            f"Padding mask length mismatch: mask_len={pad_mask.shape[0]}, batch_len={len(batch)}."
        )
    if not np.any(pad_mask):
        return

    pad_mask_torch = torch.from_numpy(pad_mask).to(device=batch.batch["response_mask"].device, dtype=torch.bool)
    batch.batch["response_mask"][pad_mask_torch] = 0


def pad_adaptive_batch_to_divisor(
    batch: DataProto,
    size_divisor: int,
    mask_key: str = ADAPTIVE_KEEP_ALL_PAD_MASK_KEY,
) -> tuple[DataProto, int]:
    """Pad adaptive batch so DP chunking can split it evenly."""
    if size_divisor <= 0:
        raise ValueError(f"size_divisor must be > 0, got {size_divisor}.")
    if mask_key in batch.non_tensor_batch:
        raise ValueError(f"Mask key '{mask_key}' already exists in non_tensor_batch.")

    original_size = len(batch)
    pad_size = (size_divisor - (original_size % size_divisor)) % size_divisor
    if pad_size == 0:
        return batch, 0

    padded_indices = _build_padding_indices(batch_size=original_size, pad_size=pad_size)
    padded_batch = batch.select_idxs(padded_indices)

    pad_mask = np.zeros(len(padded_batch), dtype=bool)
    pad_mask[-pad_size:] = True
    padded_batch.non_tensor_batch[mask_key] = pad_mask
    _apply_padding_mask_to_response_mask(padded_batch, mask_key=mask_key)
    return padded_batch, pad_size


def unpad_adaptive_batch(
    batch: DataProto,
    mask_key: str = ADAPTIVE_KEEP_ALL_PAD_MASK_KEY,
) -> tuple[DataProto, int]:
    """Remove adaptive padding rows from batch using stored mask."""
    if mask_key not in batch.non_tensor_batch:
        return batch, 0

    pad_mask = np.asarray(batch.non_tensor_batch[mask_key]).astype(bool)
    if pad_mask.shape[0] != len(batch):
        raise ValueError(
            f"Padding mask length mismatch: mask_len={pad_mask.shape[0]}, batch_len={len(batch)}."
        )
    pad_count = int(pad_mask.sum())

    if pad_count == 0:
        batch.non_tensor_batch.pop(mask_key, None)
        return batch, 0

    keep_indices = np.where(~pad_mask)[0].tolist()
    unpadded_batch = batch.select_idxs(keep_indices)
    unpadded_batch.non_tensor_batch.pop(mask_key, None)
    return unpadded_batch, pad_count
