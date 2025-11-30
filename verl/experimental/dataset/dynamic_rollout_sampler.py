from collections.abc import Iterator, Sized
from typing import List

import numpy as np
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractDynamicBatchSampler


class _DynamicBatchIterator(Iterator[List[int]]):
    """Iterator that fetches batch indices from the sampler on each __next__ call."""
    
    def __init__(self, sampler: "DynamicRolloutSampler"):
        self.sampler = sampler
    
    def __next__(self) -> list[int]:
        """Get the next batch of indices, checking sampler state each time."""
        if not self.sampler.has_more_batches():
            raise StopIteration
        
        batch = self.sampler._build_next_batch()
        if not batch:
            raise StopIteration
        
        return batch


class DynamicRolloutSampler(AbstractDynamicBatchSampler):
    """A batch sampler for dynamic rollout generation.
    
    Batches are built by:
    1. First including active problems (up to gen_batch_size)
    2. Then filling remaining slots with new problems from the shuffled pool
    
    Iteration ends when there are no more active problems AND no more new problems.
    """
    
    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.gen_batch_size = data_config.get("gen_batch_size", data_config.train_batch_size)
        self.seed = data_config.get("seed", 42)
        self.drop_last = True
        
        # Active problems that need more rollouts: UID -> dataset index
        self._active_problems: dict[str, int] = {}
        
        # Track epoch and position within epoch
        self._epoch = 0
        self._index_ptr = 0
        
        # Indices that have been used this epoch (prevents re-sampling as "new")
        self._used_indices: set[int] = set()
        
        # Shuffled indices for this epoch
        self._shuffled_indices: list[int] = []
        
        # Initialize RNG and prepare first epoch
        self._rng = np.random.RandomState(self.seed)
        self._prepare_epoch()
    
    def _prepare_epoch(self) -> None:
        """Prepare shuffled indices for a new epoch."""
        indices = list(range(len(self.data_source)))
        self._rng.shuffle(indices)
        indices = indices[:len(self) * self.gen_batch_size]
        
        self._shuffled_indices = indices
        self._index_ptr = 0
        self._used_indices.clear()
    
    def __len__(self) -> int:
        """Return upper bound on number of batches per epoch."""
        if self.drop_last:
            return len(self.data_source) // self.gen_batch_size
        else:
            return (len(self.data_source) + self.gen_batch_size - 1) // self.gen_batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Return an iterator that yields batches of indices."""
        return _DynamicBatchIterator(self)
    
    def _build_next_batch(self) -> list[int]:
        """Build the next batch of indices from active + new problems."""
        batch_indices = []
        
        # Priority 1: Active problems that need more rollouts
        active_uids = list(self._active_problems.keys())
        for uid in active_uids:
            if len(batch_indices) >= self.gen_batch_size:
                break
            batch_indices.append(self._active_problems[uid])
        
        # Priority 2: New problems from shuffled pool
        while len(batch_indices) < self.gen_batch_size:
            if self._index_ptr >= len(self._shuffled_indices):
                # No more new problems available
                break
            
            candidate_idx = self._shuffled_indices[self._index_ptr]
            self._index_ptr += 1
            
            # Skip if already used this epoch (active problems are also in used_indices)
            if candidate_idx in self._used_indices:
                continue
            
            batch_indices.append(candidate_idx)
            self._used_indices.add(candidate_idx)
        
        return batch_indices
    
    def has_more_batches(self) -> bool:
        """Check if more non-empty batches can be produced."""
        if self._active_problems:
            return True
        
        # Check for remaining unused indices
        for i in range(self._index_ptr, len(self._shuffled_indices)):
            if self._shuffled_indices[i] not in self._used_indices:
                return True
        
        return False
    
    def add_active(self, batch: DataProto) -> None:
        """Mark problems as needing more rollouts (prioritized in next batch)."""
        uids = batch.non_tensor_batch["uid"]
        indices = batch.non_tensor_batch["dataset_idx"]
        
        seen = set()
        for uid, idx in zip(uids, indices):
            if uid not in seen and uid not in self._active_problems:
                self._active_problems[uid] = idx
                self._used_indices.add(idx)
                seen.add(uid)
    
    def remove_active(self, batch: DataProto) -> None:
        """Mark problems as complete (no more rollouts needed)."""
        completed_uids = np.unique(batch.non_tensor_batch["uid"])
        for uid in completed_uids:
            self._active_problems.pop(uid, None)
    
    def on_epoch_end(self) -> None:
        """Reset state for a new epoch."""
        if self._active_problems:
            raise RuntimeError(
                f"Cannot end epoch with {len(self._active_problems)} active problems. "
                "Ensure all problems are completed before ending the epoch."
            )
        
        self._epoch += 1
        self._prepare_epoch()
    
    # StatefulDataLoader compatibility
    def state_dict(self) -> dict:
        """Return sampler state for checkpointing."""
        return {
            "epoch": self._epoch,
            "index_ptr": self._index_ptr,
            "used_indices": list(self._used_indices),
            "active_problems": dict(self._active_problems),
            "rng_state": self._rng.get_state(),
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore sampler state from checkpoint."""
        self._epoch = state_dict["epoch"]
        self._index_ptr = state_dict["index_ptr"]
        self._used_indices = set(state_dict["used_indices"])
        self._active_problems = dict(state_dict["active_problems"])
        self._rng.set_state(state_dict["rng_state"])
        
        # Regenerate shuffled indices to match checkpoint state
        epoch_rng = np.random.RandomState(self.seed)
        for _ in range(self._epoch + 1):
            indices = list(range(len(self.data_source)))
            epoch_rng.shuffle(indices)

        self._shuffled_indices = indices[:len(self) * self.gen_batch_size]
