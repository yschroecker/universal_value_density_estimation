from typing import Tuple
import numpy as np
import torch

from replay import ram_buffer, sampler
from replay.sampler import weighted_sampler


class OffsetSampler(sampler.RamReplaySampler):
    def __init__(self, buffer: ram_buffer.RamBuffer, max_offset: int, weight_column: int, cum_weight_column: int,
                 max_offset_column: int, discount_factor: float, soft_cutoff: bool=False):
        self._ram_buffer = buffer
        self._ram_buffer.add_sampler(self)
        self._weighted_sampler = weighted_sampler.WeightedSampler(buffer, weight_column, cum_weight_column)
        self._max_offset_column = max_offset_column
        self._max_offset = max_offset
        self._weight_column = weight_column
        self._soft_cutoff = soft_cutoff
        self._discount_factor = discount_factor
        self._countdown = np.arange(self._max_offset, 0, -1)

    def add_samples(self, episode_samples: np.ndarray):
        episode_samples[:-self._max_offset, self._max_offset_column] = self._max_offset
        episode_samples[-self._max_offset:, self._max_offset_column] = \
            self._countdown[-min(self._max_offset, episode_samples.shape[0]):]
        if self._soft_cutoff:
            if episode_samples.shape[0] < self._max_offset:
                episode_samples[:, self._weight_column] = 0
            else:
                try:
                    episode_samples[:-self._max_offset + 1, self._weight_column] = 1
                    episode_samples[-self._max_offset + 1:, self._weight_column] = (
                            (1 - self._discount_factor**np.arange(self._max_offset - 1, 0, -1)) /
                            (1 - self._discount_factor**self._max_offset))
                except:
                    breakpoint()
        else:
            episode_samples[:-self._max_offset + 1, self._weight_column] = 1
            episode_samples[-self._max_offset + 1:, self._weight_column] = 0

    def add_episode(self, episode_samples: np.ndarray):
        self._ram_buffer.add_samples(episode_samples)

    def sample_uniform_offset(self, batch_size: int) -> Tuple[torch.Tensor, np.ndarray]:
        start_indices = self._weighted_sampler.sample_weighted_indices(batch_size)
        max_offsets = self._ram_buffer[start_indices, self._max_offset_column]
        offsets = (np.random.rand(batch_size) * max_offsets).astype(np.int64)
        offset_indices = start_indices + offsets
        indices = np.stack([start_indices, offset_indices], axis=-1)
        samples = self._ram_buffer.load(indices)
        return samples, offsets

    def sample_discounted_offset(self, batch_size: int) -> Tuple[torch.Tensor, np.ndarray]:
        start_indices = self._weighted_sampler.sample_weighted_indices(batch_size)
        max_offsets = self._ram_buffer[start_indices, self._max_offset_column]
        offsets = (np.log(1 - np.random.rand(batch_size) * (1 - self._discount_factor**max_offsets)) /
                   np.log(self._discount_factor)).astype(np.int64)
        offset_indices = start_indices + offsets
        indices = np.stack([start_indices, offset_indices], axis=-1)
        samples = self._ram_buffer.load(indices)
        return samples, offsets

    def sample_delimited_sequence(self, batch_size: int) -> Tuple[torch.Tensor]:
        start_indices = self._weighted_sampler.sample_weighted_indices(batch_size)
        max_offsets = self._ram_buffer[start_indices, self._max_offset_column]
        offsets = (np.random.rand(batch_size) * max_offsets).astype(np.int64)
        offset_indices = start_indices + offsets
        indices = np.stack([start_indices + i for i in range(self._max_offset)], axis=-1)
        samples = self._ram_buffer.load(indices)
        return samples, offsets



class FutureStateSampler(sampler.RamReplaySampler):
    def __init__(self, buffer: ram_buffer.RamBuffer, max_offset_column: int):
        self._ram_buffer = buffer
        self._ram_buffer.add_sampler(self)
        self._max_offset_column = max_offset_column

    def add_episode(self, episode_samples: np.ndarray):
        self._ram_buffer.add_samples(episode_samples)

    def add_samples(self, episode_samples: np.ndarray):
        episode_samples[:, self._max_offset_column] = np.arange(episode_samples.shape[0], 0, -1)

    def sample_future_pair(self, batch_size: int) -> Tuple[torch.Tensor, np.ndarray]:
        start_indices = self._ram_buffer.sample_indices(batch_size)
        max_offsets = self._ram_buffer[start_indices, self._max_offset_column]
        offsets = (np.random.rand(batch_size) * max_offsets).astype(np.int64)
        offset_indices = start_indices + offsets
        indices = np.stack([start_indices, offset_indices], axis=-1)
        samples = self._ram_buffer.load(indices)
        return samples
