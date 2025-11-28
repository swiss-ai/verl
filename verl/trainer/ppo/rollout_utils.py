"""Utilities for splitting rollout batches based on completion criteria."""

import numpy as np
import torch

from verl import DataProto


def _is_problem_complete(state: dict, k_correct: int, k_incorrect: int, max_gen_budget: int) -> bool:
    """Check if a problem has met completion criteria."""
    n_incorrect = state["n_generations"] - state["n_correct"]
    has_enough_samples = state["n_correct"] >= k_correct and n_incorrect >= k_incorrect
    reached_budget = state["n_generations"] >= max_gen_budget
    return has_enough_samples or reached_budget


def split_rollouts_by_completion(
    batch: DataProto,
    problem_states: dict[str, dict],
    k_correct: int,
    k_incorrect: int,
    max_gen_budget: int,
) -> tuple[DataProto | None, DataProto | None]:
    """Split batch into problems needing more rollouts vs ready for training.
    
    Returns:
        (continue_batch, training_batch) - either may be None if empty
    """
    unique_uids, inverse = np.unique(batch.non_tensor_batch["uid"], return_inverse=True)
    
    # Check completion for each unique UID
    is_complete_per_uid = np.array([
        _is_problem_complete(problem_states.get(uid, {}), k_correct, k_incorrect, max_gen_budget)
        if uid in problem_states else False
        for uid in unique_uids
    ])
    
    # Broadcast to all rollouts and split
    is_complete = is_complete_per_uid[inverse]
    continue_idx = np.where(~is_complete)[0].tolist()
    training_idx = np.where(is_complete)[0].tolist()
    
    return (
        batch[continue_idx] if continue_idx else None,
        batch[training_idx] if training_idx else None,
    )


def update_problem_states(batch: DataProto, problem_states: dict[str, dict]) -> dict[str, dict]:
    """Update problem states based on rollout results using vectorized operations."""
    uids = np.array(batch.non_tensor_batch["uid"])
    is_correct = (batch.batch["token_level_scores"].sum(dim=-1) > 0).cpu().numpy()
    dataset_indices = batch.non_tensor_batch["index"]
    
    # Single np.unique call for all needed outputs
    unique_uids, first_indices, inverse_indices = np.unique(
        uids, return_index=True, return_inverse=True
    )
    
    # Vectorized aggregation
    counts = np.bincount(inverse_indices, minlength=len(unique_uids))
    correct_counts = np.bincount(inverse_indices, weights=is_correct, minlength=len(unique_uids))
    
    for i, uid in enumerate(unique_uids):
        state = problem_states.get(uid)
        if state is None:
            state = {
                "n_generations": 0,
                "n_correct": 0,
                "index": dataset_indices[first_indices[i]] if dataset_indices is not None else None,
            }
            problem_states[uid] = state
        
        state["n_generations"] += int(counts[i])
        state["n_correct"] += int(correct_counts[i])
    
    return problem_states


def compute_group_norm_weights(
    batch: DataProto,
    problem_states: dict[str, dict],
    n_rollouts: int,
) -> torch.Tensor:
    """Compute group normalization weights for each sample in batch.
    
    For dynamic rollout generation where different problems have different 
    numbers of rollouts, we need to normalize by group size to ensure each
    problem contributes equally to the gradient.
    
    Args:
        batch: DataProto containing the batch of samples with UIDs
        problem_states: Dictionary mapping UID -> {n_generations, n_correct, ...}
        n_rollouts: Number of rollouts per gen batch
        
    Returns:
        torch.Tensor of shape (batch_size,) with normalization weights.
        Weight = n_rollouts / n_generations for each sample.
    """
    uids = np.array(batch.non_tensor_batch["uid"])
    
    # Get unique UIDs and inverse mapping
    unique_uids, inverse_indices = np.unique(uids, return_inverse=True)
    
    # Compute weight for each unique UID once
    unique_weights = np.array([
        n_rollouts / problem_states[uid]["n_generations"]
        for uid in unique_uids
    ], dtype=np.float32)
    
    # Broadcast back to all samples
    weights = unique_weights[inverse_indices]
    
    return torch.from_numpy(weights)
