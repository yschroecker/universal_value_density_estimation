from typing import Sequence, Type, Generic, TypeVar
import abc

import copy
import torch
import numpy as np


ActionT = TypeVar('ActionT')


class Policy(Generic[ActionT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log_probability(self, states: torch.Tensor, actions: torch.Tensor) -> \
            torch.Tensor:
        pass

    @abc.abstractmethod
    def entropy(self, states: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def to(self, device: torch.device):
        pass

    @abc.abstractmethod
    def share_memory(self):
        pass

    @property
    @abc.abstractmethod
    def _module(self) -> torch.nn.Module:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abc.abstractmethod
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        pass

    @property
    @abc.abstractmethod
    def action_type(self) -> Type[np.dtype]:
        pass

    @abc.abstractmethod
    def sample_from_var(self, state: torch.Tensor, t: int=0, return_logprob: bool=False) -> ActionT:
        pass

    @abc.abstractmethod
    def mode(self, state: torch.Tensor) -> ActionT:
        pass

    def sample(self, state: np.ndarray, t: int=0, return_logprob: bool=False) -> ActionT:
        state_tensor = torch.from_numpy(np.array([state.astype(np.float32)])).to(self.device)
        return self.sample_from_var(state_tensor, t, return_logprob)

    def mode_np(self, state: np.ndarray) -> ActionT: # TODO refactor, rename
        state_tensor = torch.from_numpy(np.array([state.astype(np.float32)])).to(self.device)
        return self.mode(state_tensor)

    @abc.abstractmethod
    def clone(self):
        pass
