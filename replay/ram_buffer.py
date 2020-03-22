from typing import List, Tuple, Optional, Union

import replay.sampler
from replay import replay_buffer
import numpy as np
import torch


class RamBuffer(replay_buffer.Replay):
    def __init__(self, maxlen: int, num_features: int, device: torch.device):
        super().__init__(maxlen)
        self._maxlen = maxlen
        self._num_features = num_features
        self._device = device
        self._size = 0
        self._head = 0

        self._buffer = torch.zeros((self._maxlen, self._num_features), dtype=torch.float32).pin_memory()
        self._load_buffers = {}
        self._npbuffer = self._buffer.numpy()
        self._samplers = []

    @property
    def num_columns(self):
        return self._num_features

    @num_columns.setter
    def num_columns(self, num_features):
        self._num_features = num_features

    def load(self, indices: np.ndarray, *, index_starts_at_head=True) -> torch.Tensor:
        if index_starts_at_head:
            indices = (indices + self._head) % self._size
        return self._buffer[indices].to(self._device, non_blocking=True)

    def to_array(self) -> np.ndarray:
        return self._npbuffer[:self._size]

    def add_sampler(self, sampler):
        self._samplers.append(sampler)

    def add_samples(self, samples: Union[np.ndarray, torch.Tensor, List[np.ndarray]]):
        for sampler in self._samplers:
            sampler.add_samples(samples)
        super().add_samples(samples)

    def _add_samples_tensor(self, samples: torch.Tensor, n: int):
        self._buffer[self._head:self._head + n] = samples.detach().to(torch.device('cpu'))

    def _add_samples_array(self, samples: np.ndarray, n: int):
        self._buffer[self._head:self._head + n] = torch.from_numpy(samples)

    def _add_samples_arrays(self, samples_list: List[np.ndarray], n: int):
        column_start = 0
        for samples in samples_list:
            column_end = column_start + samples.shape[1]
            self._buffer[self._head:self._head + n, column_start:column_end] = torch.from_numpy(samples)
            column_start = column_end

    def __getitem__(self, item) -> Union[Tuple[np.ndarray, Optional[np.ndarray]], np.float32]:
        if isinstance(item, tuple):
            rows = item[0]
            columns = item[1]
        else:
            rows = item
            columns = slice(None)
        if isinstance(rows, int) or isinstance(rows, np.ndarray):
            return self._npbuffer[(rows + self._head) % self._size, columns]
        else:
            start = ((rows.start or 0) + self._head) % self._size
            stop = self._head if rows.stop is None or rows.stop >= self._size else (rows.stop + self._head) % self._size
            step = rows.step
            if stop > start or (rows.start == rows.stop and rows.start is not None):
                return self._npbuffer[start:stop:step, columns], None
            else:
                before = self._npbuffer[start::step, columns]
                if stop == 0:
                    return before, None
                else:
                    return before, self._npbuffer[:stop:step, columns]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            rows = key[0]
            columns = key[1]
        else:
            rows = key
            columns = slice(None)
        if isinstance(rows, int) or isinstance(rows, np.ndarray):
            self._npbuffer[(rows + self._head) % self._size, columns] = value
        else:
            start = ((rows.start or 0) + self._head) % self._size
            stop = self._head if rows.stop is None or rows.stop >= self._size else (rows.stop + self._head) % self._size
            step = rows.step
            if stop > start or (rows.start == rows.stop and rows.start is not None):
                self._npbuffer[start:stop:step, columns] = value
            else:
                self._npbuffer[start::step, columns] = value[:(self._size - start)]
                self._npbuffer[:stop:step, columns] = value[(self._size - start):]


def _speed_test():
    buffer = RamBuffer(10000, 500, torch.device('cuda:0'))
    buffer.add_samples(np.random.randn(10000, 500))
    import time
    start = time.time()
    print(start)
    for i in range(1000):
        #buffer.sample_sequence(100, 10)
        samples = buffer.sample(1000)
    end = time.time()
    print(end)
    print(end - start)


if __name__ == '__main__':
    _speed_test()


