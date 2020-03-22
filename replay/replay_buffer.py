from typing import Union, List
import abc
import torch
import numpy as np


class Replay(metaclass=abc.ABCMeta):
    def __init__(self, maxlen: int):
        self._maxlen = maxlen

    @property
    def size(self) -> int:
        return self._size

    @property
    @abc.abstractmethod
    def num_columns(self) -> int:
        pass

    @abc.abstractmethod
    def load(self, indices: np.ndarray) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _add_samples_tensor(self, samples: torch.Tensor, n: int):
        pass

    @abc.abstractmethod
    def _add_samples_array(self, samples: np.ndarray, n: int):
        pass

    @abc.abstractmethod
    def to_array(self) -> np.ndarray:
        pass

    def sample(self, batch_size: int) -> torch.Tensor:
        indices = np.random.choice(self._size, size=batch_size)
        return self.load(indices)

    def sample_sequence(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        start_indices = np.random.choice(self._size - sequence_length, size=batch_size)
        return self._sample_sequence_from_start_indices(start_indices, sequence_length)

    def sample_offset(self, indices: np.ndarray, offsets: np.ndarray) -> torch.Tensor:
        offset_indices = (indices + offsets) % self._size
        return offset_indices

    def sample_indices(self, batch_size: int) -> np.ndarray:
        indices = np.random.choice(self._size, size=batch_size)
        indices = (self._head + indices) % self._size
        return indices

    def add_samples(self, samples: Union[np.ndarray, torch.Tensor, List[np.ndarray]]):
        if isinstance(samples, list):
            n = samples[0].shape[0]
        else:
            n = samples.shape[0]
        old_head = self._head
        if self._head + n > self._maxlen:
            first_batch = self._maxlen - self._head
            if first_batch > 0:
                self.add_samples(samples[:first_batch])
            self._head = 0
            self.add_samples(samples[first_batch:])
        else:
            if isinstance(samples, np.ndarray):
                self._add_samples_array(samples, n)
            elif isinstance(samples, list) and isinstance(samples[0], np.ndarray):
                self._add_samples_arrays(samples, n)
            elif isinstance(samples, torch.Tensor):
                self._add_samples_tensor(samples, n)
            else:
                assert False
            self._size = min(self._size + n, self._maxlen)
            self._head += n
        self._head = self._head % self._maxlen
        return old_head, self._head

    def _sample_sequence_from_start_indices(self, start_indices: np.ndarray, sequence_length: int) -> torch.Tensor:
        indices = np.stack([start_indices + i for i in range(sequence_length)], axis=1)
        indices = (self._head + indices) % self._size
        return self.load(indices)

    def _add_samples_arrays(self, samples: List[np.ndarray], n: int):
        return self._add_samples_array(np.concatenate(samples, axis=1), n)

