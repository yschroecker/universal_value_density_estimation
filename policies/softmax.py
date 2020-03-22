from typing import Sequence, Type

import copy
import torch
import torch.nn.functional as f
import numpy as np

from policies import policy
from policies.policy import ActionT
from workflow import reporting


class SoftmaxPolicy(policy.Policy[int]):
    def __init__(self, device: torch.device, network: torch.nn.Module):
        self._network = network
        self._device = device
        reporting.register_field('max pi')

    def clone(self):
        clone_ = copy.deepcopy(self)
        clone_._network.to(self._device)
        return clone_

    def to(self, device: torch.device):
        self._network.to(device)
        self._device = device

    def share_memory(self):
        self._network.share_memory()

    @property
    def _module(self) -> torch.nn.Module:
        return self._network

    def _logits(self, states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(self._module(states))

    def log_probability(self, states: torch.Tensor, actions: torch.Tensor) -> \
            torch.Tensor:
        action_probabilities = self._logits(states)
        return action_probabilities.gather(dim=1, index=actions.long())

    def entropy(self, states: torch.Tensor) -> torch.Tensor:
        # noinspection PyArgumentList
        return -torch.sum(self._logits(states) * torch.exp(self._logits(states)), dim=1)

    def all_probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(self._module(states))

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._network.parameters()

    @property
    def action_type(self) -> Type[np.dtype]:
        return np.int32

    def sample_from_var(self, state_var: torch.Tensor, t: int=0, return_logprob: bool=False) -> int:
        probabilities = self._probabilities(state_var)
        reporting.iter_record('max pi', np.max(probabilities, keepdims=True))
        if return_logprob:
            raise NotImplementedError()
        return np.random.choice(probabilities.shape[0], p=probabilities)

    def _probabilities(self, state_var: torch.Tensor) -> np.ndarray:
        probabilities = self.all_probabilities(state_var)
        return probabilities[0].cpu().detach().numpy()

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        return self._probabilities(torch.from_numpy(state).to(self._device))

    @property
    def device(self) -> torch.device:
        return self._device

    def mode(self, state: torch.Tensor) -> int:
        probabilities = self._probabilities(state)
        return torch.argmax(probabilities)
