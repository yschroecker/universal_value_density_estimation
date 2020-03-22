import numpy as np
import torch

from replay import sampler, ram_buffer


class WeightedSampler(sampler.RamReplaySampler):
    def __init__(self, buffer: ram_buffer.RamBuffer, weight_column: int, cum_weight_column: int):
        buffer.add_sampler(self)
        self._weight_column = weight_column
        self._cum_weight_column = cum_weight_column
        self._ram_buffer = buffer

    def add_samples(self, samples: np.ndarray):
        if self._ram_buffer.size == 0:
            weight_sum = 0
        else:
            weight_sum = self._ram_buffer[-1, self._cum_weight_column]
        samples[:, self._cum_weight_column] = np.cumsum(samples[:, self._weight_column]) + weight_sum

    def sample_weighted_indices(self, batch_size: int) -> np.ndarray:
        min_weight = self._ram_buffer[0, self._cum_weight_column] - self._ram_buffer[0, self._weight_column]
        max_weight = self._ram_buffer[-1, self._cum_weight_column]
        if max_weight == min_weight:
            return np.random.choice(self._ram_buffer.size, size=batch_size)
        weight_samples = np.random.random(batch_size) * (max_weight - min_weight) + min_weight
        weight_samples = weight_samples.astype(np.float32)

        after_head, before_head = self._ram_buffer[:]
        if before_head is None:
            indices = np.searchsorted(after_head[:, self._cum_weight_column], weight_samples)
        else:
            before_head_indices = np.searchsorted(before_head[:, self._cum_weight_column], weight_samples)
            after_head_indices = np.searchsorted(after_head[:, self._cum_weight_column], weight_samples)
            head = before_head.shape[0]
            indices = np.where(after_head_indices == self._ram_buffer.size - head,
                               before_head_indices + self._ram_buffer.size - head, after_head_indices)

        return indices

    def sample_weighted(self, batch_size: int) -> torch.Tensor:
        indices = self.sample_weighted_indices(batch_size)
        return self._ram_buffer.load(indices)


if __name__ == '__main__':
    replay = ram_buffer.RamBuffer(5, 4, torch.device('cpu'))
    sampler = WeightedSampler(replay, 2, 3)
    replay.add_samples(np.array([
        [0, 0, 1, 0],
        [1, 1, 1, 0],
        [2, 2, 2, 0]
    ]))
    replay.add_samples(np.array([
        [3, 3, 0, 0],
        [4, 4, 3, 0],
    ]))

    samples = sampler.sample_weighted(10000).detach().cpu().numpy()
    print(np.bincount(samples[:, 0].astype(np.int64)))
    replay.add_samples(np.array([
        [5, 0, 1, 0],
        [6, 1, 3, 0],
    ]))
    samples = sampler.sample_weighted(10000).detach().cpu().numpy()
    print(np.bincount(samples[:, 0].astype(np.int64)))
    print(np.bincount(samples[:, 1].astype(np.int64)))

