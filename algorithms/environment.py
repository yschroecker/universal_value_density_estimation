from typing import Tuple, Generic, TypeVar, Any
import abc

import numpy as np
import gym
import gym.wrappers


ActionT = TypeVar('ActionT')


class Environment(Generic[ActionT]):
    @abc.abstractmethod
    def step(self, action: ActionT) -> Tuple[np.ndarray, float, bool, Any]:
        pass

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def state(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def needs_reset(self) -> bool:
        pass

    @abc.abstractmethod
    def cumulative_return(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def num_actions(self) -> int:
        pass


class GymEnv(Environment):
    def __init__(self, env: Any, terminal_penalty: float=0., normalize_actions: bool=False,
                 continous_action: bool=True):
        self._env = env
        self._state = None
        self._needs_reset = True
        self._terminal_penalty = terminal_penalty
        self._normalize_actions = normalize_actions
        self._continuous_action = continous_action
        self._rewards = []

    def cumulative_return(self) -> float:
        return sum(self._rewards)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def needs_reset(self) -> bool:
        return self._needs_reset

    def step(self, action: ActionT) -> Tuple[np.ndarray, float, bool, Any]:
        if not self._continuous_action and isinstance(action, np.ndarray):
            action = action[0]

        if self._normalize_actions and np.isfinite(self._env.action_space.high + self._env.action_space.low).all():
            action = (np.tanh(action)+1)/2 * (self._env.action_space.high -
                                              self._env.action_space.low) + self._env.action_space.low
        state, reward, is_terminal, info = self._env.step(action)
        self._state = state

        if is_terminal:
            self._needs_reset = True

        # workaround for gym not handling terminal states correctly
        if '_past_limit' in dir(self._env) and self._env._past_limit() or info.get('TimeLimit.truncated', False):
            is_terminal = False

        if is_terminal:
            reward += self._terminal_penalty

        self._rewards.append(reward)

        return state, reward, is_terminal, info

    def reset(self) -> np.ndarray:
        self._rewards = []
        self._needs_reset = False
        state = self._env.reset()
        self._state = state
        return state

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    @property
    def env(self) -> Any:
        return self._env

    @property
    def state_dim(self) -> int:
        return self.observation_space.shape[0]

    @property
    def action_dim(self) -> int:
        if self._continuous_action:
            return self.action_space.shape[0]
        else:
            return 1

    @property
    def num_actions(self) -> int:
        assert not self._continuous_action
        return self.action_space.n

