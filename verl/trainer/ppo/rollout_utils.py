"""
Utilities for splitting rollout batches based on completion criteria.
"""
from typing import Optional

import numpy as np

from verl import DataProto


def split_rollouts_by_completion(
    batch: DataProto,
    problem_states: dict[str, dict],
    k_correct: int,
    k_incorrect: int,
    max_gen_budget: int,
) -> tuple[Optional[DataProto], Optional[DataProto]]:
    """Split rollout batch into problems that need more rollouts vs ready for training.
    
    This function examines each problem in the batch and categorizes it based on
    completion criteria. Problems are split into two groups:
    1. Those needing more rollouts (not yet meeting criteria)
    2. Those ready for training (meeting k_correct or max_gen_budget)
    
    Args:
        batch: DataProto containing rollout results
        problem_states: Dictionary mapping UID to problem state containing:
            - n_generations: Total generations so far
            - n_correct: Number of correct solutions
        k_correct: Minimum correct solutions required
        k_incorrect: Minimum incorrect solutions allowed
        max_gen_budget: Maximum generations allowed per problem
    
    Returns:
        Tuple of (continue_generation_batch, training_ready_batch)
        - continue_generation_batch: Rollouts needing more generations (or None)
        - training_ready_batch: Rollouts ready for training (or None)
    """
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("Batch must contain 'uid' in non_tensor_batch")
    
    unique_uids, inverse_indices = np.unique(batch.non_tensor_batch["uid"], return_inverse=True)
    
    # Determine completion status for each unique UID
    unique_is_complete = []
    for uid in unique_uids:
        if uid in problem_states:
            state = problem_states[uid]
            unique_is_complete.append(
                (state["n_correct"] >= k_correct and state["n_generations"] - state["n_correct"] >= k_incorrect)
                or state["n_generations"] >= max_gen_budget
            )
            # TODO JUAN: logs
            if state["n_generations"] > max_gen_budget:
                print(f"DEBUG: UID {uid} reached max_gen_budget {max_gen_budget} with n_generations {state['n_generations']}")
        else:
            unique_is_complete.append(False)
            
    unique_is_complete = np.array(unique_is_complete, dtype=bool)
            
    # Broadcast status back to all rollouts
    is_complete = unique_is_complete[inverse_indices]

    # Get indices
    continue_indices = np.where(~is_complete)[0].tolist()
    training_indices = np.where(is_complete)[0].tolist()
    
    # Create split batches
    continue_batch = batch[continue_indices] if continue_indices else None
    training_batch = batch[training_indices] if training_indices else None
    
    return continue_batch, training_batch


def update_problem_states(
    batch: DataProto,
    problem_states: dict[str, dict],
) -> dict[str, dict]:
    """Update problem states based on rollout results using vectorized operations.
    
    Args:
        batch: DataProto containing rollout results with:
            - batch["token_level_scores"]: Token-level scores per rollout
            - non_tensor_batch["uid"]: Problem UIDs
            - non_tensor_batch["index"]: Original dataset index
        problem_states: Existing problem states dictionary to update
    
    Returns:
        Updated problem_states dictionary
    """
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("Batch must contain 'uid' in non_tensor_batch for tracking")
    
    uids = np.array(batch.non_tensor_batch["uid"])
    is_correct = (batch.batch["token_level_scores"].sum(dim=-1) > 0).cpu().numpy()
    dataset_indices = batch.non_tensor_batch["index"]
    
    # Get unique UIDs and inverse indices to aggregate counts
    unique_uids, inverse_indices = np.unique(uids, return_inverse=True)
    
    # Use bincount for O(N) aggregation instead of O(N*M) loop
    counts = np.bincount(inverse_indices, minlength=len(unique_uids))
    correct_counts = np.bincount(inverse_indices, weights=is_correct, minlength=len(unique_uids))
    
    # Get first occurrence indices to retrieve dataset_indices for new problems
    _, first_indices = np.unique(uids, return_index=True)
    
    # Update each problem's state
    for i, uid in enumerate(unique_uids):
        count = counts[i]
        n_correct = correct_counts[i]
        
        if uid not in problem_states:
            # Initialize new problem
            problem_states[uid] = {
                "n_generations": 0,
                "n_correct": 0,
                "index": dataset_indices[first_indices[i]] if dataset_indices is not None else None,
            }
        
        state = problem_states[uid]
        state["n_generations"] += int(count)
        state["n_correct"] += int(n_correct)
    
    return problem_states
