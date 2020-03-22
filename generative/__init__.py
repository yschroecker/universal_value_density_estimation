from typing import Optional
import abc

import torch


class ImplicitGenerativeModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_samples(self, batch_size: int, context: Optional[torch.Tensor]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def train(self, iteration: int, true_samples: torch.Tensor, context: Optional[torch.Tensor]):
        pass

