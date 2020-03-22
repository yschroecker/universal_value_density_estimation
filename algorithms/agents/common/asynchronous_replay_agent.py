from typing import Callable, Union, NamedTuple, Tuple
import abc

import numpy as np
import torch
import torch.multiprocessing
import time
from concurrent import futures


from algorithms import environment, data
from algorithms.agents.common.synchronous_replay_agent import ReplayAgentParams
from replay import ram_buffer, replay_description
from replay.replay_description import ReplayColumn as RC
from workflow import reporting
from policies import policy
import sys
import logging

class AsyncReplayAgentParams(ReplayAgentParams, NamedTuple):
    replay_size: int
    min_replay_size: int
    batch_size: int
    sequence_length: int
    buffer_type: str = 'ram'
    num_envs: int = 1
    actor_device: str = 'cpu'
    step_limit: float = 0.


class AsynchronousReplayAgentBase(metaclass=abc.ABCMeta):
    def __init__(self, make_env: Callable[[], environment.Environment[Union[np.ndarray, int]]], device: torch.device,
                 params: AsyncReplayAgentParams):
        self._params = params
        self._device = device
        self._actor_device = torch.device(self._params.actor_device)
        self._executor = futures.ProcessPoolExecutor(max_workers=self._params.num_envs,
                                                     mp_context=torch.multiprocessing.get_context('spawn'),
                                                     initializer=self._initialize_process_env,
                                                     initargs=(make_env,))
        self._futures = None

        self._make_env = make_env
        self._env = make_env()
        self._state_dim = self._env.state_dim
        self._action_dim = self._env.action_dim

        self._replay_description = self.get_description(self._env)
        self._buffer = ram_buffer.RamBuffer(self._params.replay_size, self._replay_description.num_columns, device)

        reporting.register_field("return")
        self._env_steps = 0
        reporting.register_field("env_steps")
        self.__update_count = 0

    @abc.abstractmethod
    def get_description(self, env: environment.Environment):
        pass

    @abc.abstractmethod
    def freeze_policy(self, device: torch.device) -> policy.Policy:
        pass

    @staticmethod
    def _initialize_process_env(make_env):
        global _process_env
        _process_env = make_env()
        _process_env.reset()

    @classmethod
    def _collect_trajectory(cls, replay_description_: replay_description.ReplayDescription,
                            frozen_policy: policy.Policy):
        new_transitions = {}
        while not _process_env.needs_reset:
            state = _process_env.state
            action, action_logprob = frozen_policy.sample(state, return_logprob=True)

            next_state, reward, is_terminal, info = _process_env.step(action)
            is_timeout = _process_env.needs_reset
            terminal_weight = 0. if is_terminal else 1.
            timeout_weight = 0. if is_timeout else 1.
            new_transition = {'states': state,
                              'actions': action,
                              'rewards': reward,
                              'next_states': next_state,
                              'timeout_weight': timeout_weight,
                              'terminal_weight': terminal_weight,
                              'action_log_prob': action_logprob,
                              **info}
            for key in new_transition:
                if key not in new_transitions:
                    new_transitions[key] = [new_transition[key]]
                else:
                    new_transitions[key].append(new_transition[key])

        new_transitions = replay_description_.prepare_samples((len(new_transitions['states']),), new_transitions)

        cumulative_return = _process_env.cumulative_return()
        _process_env.reset()
        del frozen_policy
        return new_transitions, cumulative_return

    def update(self):
        logging.getLogger().setLevel(logging.DEBUG)
        self.__update_count += 1
        if self._futures is None:
            self._futures = []
            for i in range(self._params.num_envs):
                frozen_policy = self.freeze_policy(self._actor_device)
                frozen_policy.share_memory()
                self._futures.append(self._executor.submit(self._collect_trajectory, self._replay_description,
                                                           frozen_policy))
        else:
            for i in range(self._params.num_envs):
                if self._futures[i].done() and (self._env_steps - self._params.min_replay_size) * self._params.step_limit < self.__update_count:
                    frozen_policy = self.freeze_policy(self._actor_device)
                    frozen_policy.share_memory()
                    new_trajectory, cumulative_return = self._futures[i].result()
                    self._futures[i] = self._executor.submit(self._collect_trajectory, self._replay_description,
                                                             frozen_policy)
                    self._add_episode(new_trajectory)

                    self._env_steps += new_trajectory.shape[0]
                    reporting.iter_record("return", cumulative_return)

        reporting.iter_record("env_steps", self._env_steps)
        if self._buffer.size >= self._params.min_replay_size:
            self._update()
        else:
            time.sleep(0.1)

    def _add_episode(self, new_trajectory: data.TransitionSequence):
        self._buffer.add_samples(new_trajectory)

    @abc.abstractmethod
    def sample_action(self, state: np.ndarray) -> Tuple[Union[np.ndarray, int], float]:
        pass

    @abc.abstractmethod
    def _update(self):
        pass


class AsynchronousReplayAgent(AsynchronousReplayAgentBase, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_description(self, env):
        return replay_description.ReplayDescription([
            RC("states", self._state_dim),
            RC("actions", self._action_dim),
            RC("rewards", 0),
            RC("next_states", self._state_dim),
            RC("timeout_weight", 0),
            RC("terminal_weight", 0),
            RC("action_log_prob", 0),
        ], data.TransitionSequence)

