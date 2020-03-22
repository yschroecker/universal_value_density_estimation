from typing import Any, Sequence
import abc
import copy
import torch
from algorithms import data
import util


class TemporalDifferenceBase(metaclass=abc.ABCMeta):
    def __init__(self, model: torch.nn.Module, target_update_rate: int, target_update_step: float=1):
        self._online_network = model
        self._target_update_rate = target_update_rate
        if self._target_update_rate == 1 and target_update_step == 1:
            self._target_network = lambda *args: self._online_network(*args).detach()
        else:
            #self._target_network = copy.deepcopy(self._online_network).to(next(self._online_network.parameters()).device)
            self._target_network = util.target_network(self._online_network)

        self._update_counter = 0
        self.name = ""
        self._target_update_step = target_update_step

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._online_network.parameters()

    def update_target(self):
        if self._target_update_rate > 1:
            if self._target_update_step == 1:
                self._target_network = copy.deepcopy(self._online_network).to(
                    next(self._online_network.parameters()).device)
            else:
                for online_param, target_param in zip(self._online_network.parameters(), self._target_network.parameters()):
                    target_param.requires_grad = False
                    target_param.data = ((1 - self._target_update_step) * target_param +
                                        self._target_update_step * online_param).detach()

    def update_loss(self, batch: Any, *args, **kwargs) -> torch.Tensor:
        loss = self._update(batch, *args, **kwargs)

        # noinspection PyArgumentList
        self._update_counter += 1
        if self._target_update_rate > 1 and self._update_counter % self._target_update_rate == 0:
            self.update_target()
        return loss

    @abc.abstractmethod
    def _update(self, batch: Any, *args, **kwargs) -> torch.Tensor:
        pass


class ValueTemporalDifferenceBase(TemporalDifferenceBase):
    @abc.abstractmethod
    def values(self, states: torch.Tensor) -> torch.Tensor:
        pass
