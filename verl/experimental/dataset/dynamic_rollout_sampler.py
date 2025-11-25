"""
Dynamic rollout sampler that adaptively generates rollouts per problem based on completion criteria.

This sampler maintains state about which problems need more rollouts and yields problems
accordingly, seamlessly integrating with PyTorch's DataLoader infrastructure.
"""

from collections.abc import Iterator, Sized

import numpy as np
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractDynamicSampler


class DynamicRolloutSampler(AbstractDynamicSampler):
    """Sampler that dynamically controls which problems receive more rollouts.
    
    This sampler tracks completion criteria per problem (e.g., k correct solutions or 
    generation budget exhausted) and maintains a pool of problems that need more rollouts.
    
    Key Features:
    - Tracks per-problem state (number of generations, correct count)
    - Automatically refills from base dataset when problems complete
    - Handles both initial sampling and carry-over from previous batches
    - Clean separation from training logic
    
    Args:
        data_source: The underlying dataset to sample from
        data_config: Configuration containing:
            - gen_batch_size: Number of problems per generation batch
            - seed: Random seed for reproducibility
    """
    
    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source, data_config)
        
        self.data_source = data_source
        self.gen_batch_size = data_config.get("gen_batch_size", data_config.train_batch_size)
        
        # Active problems that need more rollouts (UID -> dataset index)
        self.active_problems: dict[str, int] = {}
        
        # Track all indices that have been sampled this epoch (to prevent re-sampling)
        self.sampled_indices: set[int] = set()
        
        # RNG for reproducibility
        self.rng = np.random.RandomState(data_config.get("seed", 42))
        
        # Track epoch for proper iteration
        self.epoch = 0
        self.index_ptr = 0

        # Compute available indices for sampling
        indices = list(range(len(data_source)))
        self.rng.shuffle(indices)
        
        # Handle dropping last
        self.drop_last = True
        self.available_indices = indices[:len(self)]
        
    def __len__(self) -> int:
        """Return the upper-bound of samples that will be yielded."""
        if self.drop_last:
            return (len(self.data_source) // self.gen_batch_size) * self.gen_batch_size
        else:
            return len(self.data_source)
    
    def __iter__(self) -> Iterator[int]:
        """Yield dataset indices for one logical batch.
        
        Each call to __iter__ yields up to gen_batch_size indices:
        - First, active problems that need more rollouts
        - Then, new problems from the available pool
        
        After yielding one batch, the iterator stops. The trainer should
        call add_active/remove_active to update state, then the DataLoader
        will create a new iterator for the next batch.
        
        This design ensures active problems can be re-sampled across batches
        while preventing duplicates within a single batch.
        """
        batch_indices = self.get_next_batch_indices()
        
        # Yield all indices for this batch
        for idx in batch_indices:
            yield idx
        
        # Iterator stops here - trainer must update state and get new iterator
    
    def get_next_batch_indices(self) -> list[int]:
        """Get the next batch of indices directly (bypasses iterator protocol).
        
        This method provides direct access to batch indices without going through
        the iterator protocol, which is useful for dynamic sampling scenarios.
        
        Returns:
            List of dataset indices for the next batch (may be empty if epoch complete)
        """
        batch_indices = []
        
        # Priority 1: Include active problems that need more rollouts
        active_uids = list(self.active_problems.keys())
        for uid in active_uids[:self.gen_batch_size]:
            batch_indices.append(self.active_problems[uid])
        
        # Priority 2: Fill remaining slots with new problems
        while len(batch_indices) < self.gen_batch_size and self.index_ptr < len(self.available_indices):
            candidate_idx = self.available_indices[self.index_ptr]
            self.index_ptr += 1
            # Skip indices already sampled this epoch
            if candidate_idx not in self.sampled_indices:
                batch_indices.append(candidate_idx)
                self.sampled_indices.add(candidate_idx)
        
        return batch_indices
    
    def has_next_batch(self) -> bool:
        """Check if the sampler can yield another non-empty batch.
        
        Returns:
            True if there are active problems or unsampled indices that would
            result in a non-empty batch
        """
        # Check if there are active problems to yield
        if len(self.active_problems) > 0:
            return True
        
        # Check if there are unsampled indices remaining
        for i in range(self.index_ptr, len(self.available_indices)):
            if self.available_indices[i] not in self.sampled_indices:
                return True
        
        return False
    
    def num_available_new_problems(self) -> int:
        """Count how many new (unsampled) problems are available.
        
        Returns:
            Number of indices that haven't been sampled yet this epoch
        """
        count = 0
        for i in range(self.index_ptr, len(self.available_indices)):
            if self.available_indices[i] not in self.sampled_indices:
                count += 1
        return count
    
    def add_active(self, batch: DataProto) -> None:
        """Add problems to the active pool that need more rollouts.
        
        Args:
            batch: DataProto containing problems that need more generations
        """
        uids = batch.non_tensor_batch["uid"]
        indices = batch.non_tensor_batch["index"]
        
        unique_uids = np.unique(uids)
        for uid in unique_uids:
            if uid not in self.active_problems:
                idx = indices[uids == uid][0]
                self.active_problems[uid] = idx
                # Mark as sampled so it won't be picked again from available_indices
                self.sampled_indices.add(idx)
    
    def remove_active(self, batch: DataProto) -> None:
        """Remove completed problems from the active pool.
        
        Args:
            batch: DataProto containing completed problems
        """
        completed_uids = np.unique(batch.non_tensor_batch["uid"])
        for uid in completed_uids:
            if uid in self.active_problems:
                del self.active_problems[uid]
    
    def on_epoch_end(self) -> None:
        """Reset for a new epoch."""
        assert len(self.active_problems) == 0, "Cannot reset epoch while still having active problems."
        self.epoch += 1
        self.index_ptr = 0
        self.sampled_indices.clear()  # Reset sampled tracking for new epoch
        
        indices = list(range(len(self.data_source)))
        self.rng.shuffle(indices)
        self.available_indices = indices[:len(self)]
