import abc
import numpy as np
from replay import ram_buffer


class RamReplaySampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_samples(self, samples: np.ndarray):
        pass
