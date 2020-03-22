from typing import Callable, Union, NamedTuple, Tuple
import abc

import numpy as np
import torch

from algorithms import environment, data
from replay import ram_buffer, vram_buffer
from workflow import reporting


class ReplayAgentParams(NamedTuple):
    replay_size: int
    min_replay_size: int
    batch_size: int
    sequence_length: int
    buffer_type: str = 'ram'
    steps_per_update: int = 1


class SynchronousReplayAgent:
    def __init__(self, make_env: Callable[[], environment.Environment[Union[np.ndarray, int]]], device: torch.device,
                 params: ReplayAgentParams):
        self._params = params
        self._env = make_env()
        self._env.reset()
        self._state_dim = self._env.state_dim
        self._action_dim = self._env.action_dim
        self._device = device

        # state, action, reward, next_state, timeout, terminal, action_logprob
        buffer_dim = 2 * self._state_dim + self._action_dim + 4
        if params.buffer_type == 'ram':
            self._buffer = ram_buffer.RamBuffer(self._params.replay_size, buffer_dim, device)
        elif params.buffer_type == 'vram':
            self._buffer = vram_buffer.VramBuffer(self._params.replay_size, buffer_dim, device)
        else:
            assert False

        reporting.register_field("return")

    @staticmethod
    def _to_buffer_format(state, action, reward, next_state, timeout, terminal, action_logprob):
        return np.concatenate([state, np.atleast_1d(action), np.atleast_1d(reward), next_state, np.atleast_1d(timeout),
                               np.atleast_1d(terminal), np.atleast_1d(action_logprob)], axis=0)[None]

    def _buffer_to_sequence(self, buffer_object):
        state_idx = 0
        action_idx = state_idx + self._state_dim
        reward_idx = action_idx + self._action_dim
        next_state_idx = reward_idx + 1
        timeout_idx = next_state_idx + self._state_dim
        terminal_idx = timeout_idx + 1
        log_prob_idx = terminal_idx + 1

        state: torch.Tensor = buffer_object[:, :, state_idx:action_idx]
        action: torch.Tensor = buffer_object[:, :, action_idx:reward_idx]
        reward: torch.Tensor = buffer_object[:, :, reward_idx]
        next_state: torch.Tensor = buffer_object[:, :, next_state_idx:timeout_idx]
        timeout: torch.Tensor = buffer_object[:, :, timeout_idx]
        terminal: torch.Tensor = buffer_object[:, :, terminal_idx]
        log_prob: torch.Tensor = buffer_object[:, :, log_prob_idx]
        return data.TransitionSequence(
            states=state, actions=action, rewards=reward, next_states=next_state, timeout_weight=timeout,
            terminal_weight=terminal, action_log_prob=log_prob
        )

    def update(self):
        for t in range(self._params.steps_per_update):
            state = self._env.state
            action, action_logprob = self.sample_action(state)

            next_state, reward, is_terminal, _ = self._env.step(action)
            is_timeout = self._env.needs_reset
            terminal_weight = 0. if is_terminal else 1.
            timeout_weight = 0. if is_timeout else 1.

            self._buffer.add_samples(self._to_buffer_format(state, action, reward, next_state,
                                                            timeout_weight, terminal_weight, action_logprob))

            if self._env.needs_reset:
                reporting.iter_record("return", self._env.cumulative_return())
                self._env.reset()

        if self._buffer.size >= self._params.min_replay_size:
            sequence = self._buffer.sample_sequence(self._params.batch_size, self._params.sequence_length)
            self._update(self._buffer_to_sequence(sequence))

    @abc.abstractmethod
    def sample_action(self, state: np.ndarray) -> Tuple[Union[np.ndarray, int], float]:
        pass

    @abc.abstractmethod
    def _update(self, batch: data.TransitionSequence):
        pass

