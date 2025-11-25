# Copyright 2025 Amazon.com Inc and/or its affiliates
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
from abc import abstractmethod
from collections.abc import Sized

from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass


class AbstractDynamicSampler(AbstractSampler):
    """Abstract interface for dynamic samplers that track active problems.
    
    This sampler supports adaptive problem selection where problems can be:
    - Added to the active pool when they need more work
    - Removed from the active pool when they complete
    """

    @abstractmethod
    def add_active(self, batch: DataProto) -> None:
        """Add problems to the active pool that need more rollouts.
        
        Args:
            batch: DataProto containing problems to mark as active
        """
        pass

    @abstractmethod
    def remove_active(self, batch: DataProto) -> None:
        """Remove completed problems from the active pool.
        
        Args:
            batch: DataProto containing problems to remove from active pool
        """
        pass

    @abstractmethod
    def get_next_batch_indices(self) -> list[int]:
        """Get the next batch of indices directly (bypasses iterator protocol).
        
        This method provides direct access to batch indices without going through
        the iterator protocol, which is useful for dynamic sampling scenarios
        where the sampler state changes between batches.
        
        Returns:
            List of dataset indices for the next batch (may be empty if complete)
        """
        pass

    @abstractmethod
    def has_next_batch(self) -> bool:
        """Check if the sampler can yield another non-empty batch.
        
        Returns:
            True if a call to get_next_batch_indices() would return a non-empty list
        """
        pass
