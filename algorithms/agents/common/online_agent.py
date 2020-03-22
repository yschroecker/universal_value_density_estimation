from typing import Callable,NamedTuple, Union
import abc

from algorithms import environment
from workflow import reporting
from algorithms import data

import numpy as np
import torch


class OnlineAgentParams(NamedTuple):
    batch_size: int
    discount_factor: float
    num_envs: int = 1
    num_steps: int = 1


class OnlineAgent:
    def __init__(self, make_env: Callable[[], environment.Environment[Union[np.ndarray, int]]], device: torch.device,
                 params: OnlineAgentParams):
        self._params = params
        self._envs = [make_env() for _ in range(params.num_envs)]
        state_dim = self._envs[0].state_dim
        action_dim = self._envs[0].action_dim
        self._device = device

        self._states = torch.zeros((params.batch_size, state_dim)).pin_memory()
        self._actions = torch.zeros((params.batch_size, action_dim)).pin_memory()
        self._rewards = torch.zeros((params.batch_size,)).pin_memory()
        self._bootstrap_weights = torch.zeros((params.batch_size,)).pin_memory()
        self._bootstrap_states = torch.zeros((params.batch_size, state_dim)).pin_memory()
        self._bootstrap_actions = torch.zeros((params.batch_size, action_dim)).pin_memory()

        reporting.register_field("return")

    def update(self):
        i = 0
        while i < self._params.batch_size:
            env = np.random.choice(self._envs)
            j = i
            for t in range(self._params.num_steps):
                if env.needs_reset:
                    reporting.iter_record("return", env.cumulative_return())
                    env.reset()
                    break

                state = env.state
                action = np.atleast_1d(self.sample_action(state))
                next_state, reward, is_terminal, _ = env.step(action)

                if is_terminal:
                    self._bootstrap_weights[i] = 0.
                else:
                    self._bootstrap_weights[i] = self._params.discount_factor
                self._rewards[i] = reward
                self._states[i] = torch.Tensor(state)
                self._actions[i] = torch.Tensor(action)
                self._bootstrap_states[i] = torch.Tensor(next_state)

                for past in range(j, i):
                    self._rewards[past] += reward * self._params.discount_factor ** (i + 1 - past)
                    self._bootstrap_states[past] = self._bootstrap_states[i]
                    self._bootstrap_weights[past] *= self._bootstrap_weights[i]

                i += 1

                if i >= self._params.batch_size:
                    break

        td_batch = data.TDBatch(
            states=self._states.to(self._device),
            actions=self._actions.to(self._device),
            intermediate_returns=self._rewards.to(self._device),
            bootstrap_weights=self._bootstrap_weights.to(self._device),
            bootstrap_states=self._bootstrap_states.to(self._device),
            bootstrap_actions=self._bootstrap_actions.to(self._device))

        self._update(td_batch)

    @abc.abstractmethod
    def sample_action(self, state: np.ndarray) -> Union[np.ndarray, int]:
        pass

    @abc.abstractmethod
    def _update(self, batch: data.TDBatch):
        pass
