from replay import replay_buffer
import numpy as np
import torch


class VramBuffer(replay_buffer.Replay):
    def __init__(self, maxlen: int, num_features: int, device: torch.device):
        super().__init__(maxlen)
        self._buffer = torch.zeros((maxlen, num_features), dtype=torch.float32).to(device)
        self._device = device
        self._last_n = -1
        self._num_features = num_features
        self._size = 0
        self._head = 0

    @property
    def num_columns(self):
        return self._buffer.shape[1]

    def load(self, indices: np.ndarray) -> torch.Tensor:
        return self.__getitem__(indices)

    def to_array(self) -> np.ndarray:
        return self._buffer[:self._size].cpu().detach().numpy()

    def _add_samples_tensor(self, samples: torch.Tensor, n: int):
        self._buffer[self._head:self._head + n] = samples.detach().to(self._device, non_blocking=True, copy=False)

    def _add_samples_array(self, samples: np.ndarray, n: int):
        samples = torch.from_numpy(samples).pin_memory()
        self._buffer[self._head:self._head + n] = samples.to(self._device, non_blocking=True)

    def __getitem__(self, item) -> torch.Tensor:
        if isinstance(item, tuple):
            rows = item[0]
            columns = item[1]
        else:
            rows = item
            columns = slice(None)
        assert isinstance(rows, int) or isinstance(rows, np.ndarray)
        return self._buffer[(rows + self._head) % self._size, columns]


def _run():
    buffer = VramBuffer(5, 2, device=torch.device('cpu'))
    buffer.add_samples(np.array([[1, 1], [2, 2]]))
    buffer.add_samples(np.array([[3, 3], [4, 4]]))
    buffer.add_samples(np.array([[5, 5], [6, 6]]))
    print(buffer._buffer)
    buffer.add_samples(np.array([[7, 7], [8, 8]]))
    buffer.add_samples(np.array([[9, 9], [10, 10]]))
    print(buffer._buffer)
    buffer.add_samples(np.array([[11, 11]]))
    print(buffer._buffer)

def _speed_test():
    buffer = VramBuffer(10000, 5000, torch.device('cuda:0'))
    buffer.add_samples(np.random.randn(10000, 5000))
    import time
    start = time.time()
    print(start)
    for i in range(1000):
        buffer.sample_sequence(100, 10)
    end = time.time()
    print(end)
    print(end - start)

if __name__ == '__main__':
    _speed_test()
