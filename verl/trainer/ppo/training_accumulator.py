"""
Training accumulator for dynamic rollout generation.

This module provides a clean interface for accumulating completed rollouts until
a full training batch is ready, then triggering gradient updates.
"""

from typing import Optional
from collections import deque

import numpy as np

from verl import DataProto


class TrainingAccumulator:
    """Accumulates rollout data for training until batch size is reached.
    
    This class manages TWO types of rollouts:
    1. Incomplete rollouts: Problems that haven't met criteria yet (stored by UID)
    2. Complete rollouts: Problems ready for training (buffered for batch assembly)
    
    Key Features:
    - Stores incomplete rollouts per-problem for future completion
    - Accumulates complete rollouts for training batches
    - Efficient DataProto concatenation
    - Automatic batch size management
    - Memory-efficient with configurable limits
    
    Args:
        train_batch_size: Target size for training batches
    """
    
    def __init__(self, train_batch_size: int):
        self.train_batch_size = train_batch_size
        
        # Buffer of complete rollouts awaiting training (dict by UID for fair sampling)
        self.training_buffer: dict[str, DataProto] = {}
        self._uid_queue: deque[str] = deque()
        
        # Storage for incomplete rollouts by problem UID
        self.incomplete_rollouts_buffer: dict[str, DataProto] = {}
    
    def add_incomplete(self, data: DataProto) -> None:
        """Store incomplete rollouts (problems that haven't met criteria yet).
        
        Groups rollouts by UID and stores them for future completion.
        
        Args:
            data: DataProto containing rollouts that need more generations
        """
        if data is None or len(data) == 0:
            return
        
        # Group by UID
        uids = data.non_tensor_batch["uid"]
        unique_uids = np.unique(uids)
        
        for uid in unique_uids:
            uid_indices = np.where(uids == uid)[0].tolist()
            
            if uid not in self.incomplete_rollouts_buffer:
                self.incomplete_rollouts_buffer[uid] = data.select_idxs(uid_indices)
            else:
                self.incomplete_rollouts_buffer[uid] = DataProto.concat([self.incomplete_rollouts_buffer[uid], data.select_idxs(uid_indices)])

    def add_complete(self, data: DataProto) -> None:
        """Move incomplete rollouts for completed problems to training buffer.
        
        When a problem meets criteria, all its accumulated rollouts are moved
        to the training buffer.
        
        Args:
            data: DataProto containing rollouts ready for training
        """
        if data is None or len(data) == 0:
            return
        
        # Group by UID
        uids = data.non_tensor_batch["uid"]
        unique_uids = np.unique(uids)
        
        for uid in unique_uids:
            uid_indices = np.where(uids == uid)[0]
            uid_data = data.select_idxs(uid_indices)
            
            if uid in self.training_buffer:
                self.training_buffer[uid] = DataProto.concat([self.training_buffer[uid], uid_data])
            elif uid in self.incomplete_rollouts_buffer:
                old_data = self.incomplete_rollouts_buffer.pop(uid)
                self.training_buffer[uid] = DataProto.concat([old_data, uid_data])
                self._uid_queue.append(uid)
            else:
                self.training_buffer[uid] = uid_data
                self._uid_queue.append(uid)

    def get_training_batch(self) -> Optional[DataProto]:
        """Get a training batch if enough problems are accumulated.
        
        Returns:
            Training batch with train_batch_size problems, or None if not enough data
        """
        if len(self.training_buffer) >= self.train_batch_size:
            return self._flush()
        return None
    
    def _flush(self) -> Optional[DataProto]:
        """Concatenate buffered data into a training batch using FIFO order.
        
        Returns exactly train_batch_size problems, keeping the rest for next batch.
        Uses FIFO queue to ensure older problems are trained first.
        
        Returns:
            Training batch with train_batch_size problems, or None if buffer empty
        """
        if not self.training_buffer:
            return None
        
        batch_problems = []
        while len(batch_problems) < self.train_batch_size and self._uid_queue:
            uid = self._uid_queue.popleft()
            problem_data = self.training_buffer.pop(uid)
            batch_problems.append(problem_data)
        
        return DataProto.concat(batch_problems)
    
    def on_epoch_end(self) -> None:
        """Clean up and validate accumulator state at epoch end.
        
        Ensures all problems have been completed and no data leaks between epochs.
        Raises AssertionError if buffers are not empty.
        """
        assert len(self.training_buffer) == 0, \
            f"Training buffer not empty at epoch end: {len(self.training_buffer)} problems remaining"
        assert len(self.incomplete_rollouts_buffer) == 0, \
            f"Incomplete rollouts buffer not empty at epoch end: {len(self.incomplete_rollouts_buffer)} problems remaining"
        
        # Reset internal counters for safety
        self._uid_queue.clear()
