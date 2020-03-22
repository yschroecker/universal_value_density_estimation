from typing import Tuple, Any
from algorithms import environment
from algorithms.environment import ActionT
import numpy as np


class NormalizedEnv(environment.Environment):
    def step(self, action: ActionT) -> Tuple[np.ndarray, float, bool, Any]:
        self._t += 1
        state, reward, is_terminal, info = self._gym_env.step(action)
        info['unnormalized_state'] = state
        state = self.normalize_state(state)
        return state, reward, is_terminal, info

    def reset(self) -> np.ndarray:
        self._t = 0
        state = self.normalize_state(self._gym_env.reset())
        return state

    @property
    def state(self) -> np.ndarray:
        return self.normalize_state(self._gym_env.state)

    @property
    def needs_reset(self) -> bool:
        return self._gym_env.needs_reset

    def cumulative_return(self) -> float:
        return self._gym_env.cumulative_return()

    @property
    def state_dim(self) -> int:
        return self._relevant_states.sum()

    @property
    def action_dim(self) -> int:
        return self._gym_env.action_dim

    @property
    def num_actions(self) -> int:
        return self._gym_env.num_actions

    def __init__(self, env_builder, state_min: np.ndarray, state_max: np.ndarray):
        self._gym_env = env_builder()
        self._relevant_states = state_max > state_min
        self._state_min = state_min[self._relevant_states]
        self._state_max = state_max[self._relevant_states]

    def normalize_state(self, state):
        if state.ndim == 1:
            state_min = self._state_min
            state_max = self._state_max
            state = state[self._relevant_states]
        else:
            state_min = self._state_min[None, :]
            state_max = self._state_max[None, :]
            state = state[:, self._relevant_states]
        state = (state - state_min)/(state_max - state_min) * 2 - 1

        if state.ndim == 1:
            state = np.concatenate([state[::2], state[1::2]])
        else:
            state = np.concatenate([state[:, ::2], state[:, 1::2]], axis=1)

        return state

class ActionEnv(environment.GymEnv):
    def __init__(self, env):
        super().__init__(env)

    @property
    def state_dim(self) -> int:
        return self.observation_space.shape[0] + self.action_dim

    def reset(self):
        base_state = super().reset()
        self._state = np.concatenate([base_state, np.zeros((self.action_dim,))])
        return self._state

    def step(self, action):
        base_state, reward, is_terminal, info = super().step(action)
        self._state = np.concatenate([base_state, action])
        return self._state, reward, is_terminal, info
