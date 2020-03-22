import gym
from gym import core, spaces
from gym.envs.mujoco import humanoid
from gym.envs import registration
import numpy as np
from distutils import version

assert version.StrictVersion(gym.__version__) >= "0.14.0"

class NVEnv(core.Env):
    def __init__(self, base_env: str, velocity_idx: int, terminate_on_fall: bool, exclude_y_velocity: bool=False,
                 alive_reward: float=0.):
        self._alive_reward = alive_reward
        self._exclude_y_velocity = exclude_y_velocity
        self._env = gym.make(base_env)
        self._velocity_idx = velocity_idx
        low = self.transform_state(self._env.observation_space.low)
        high = self.transform_state(self._env.observation_space.high)
        self.observation_space = spaces.Box(low, high)
        self._terminate_on_fall = terminate_on_fall

    def transform_state(self, state):
        if self._exclude_y_velocity:
            state = np.concatenate([state[..., :self._velocity_idx], state[..., self._velocity_idx+2:]], axis=-1)
        else:
            state = np.concatenate([state[..., :self._velocity_idx], state[..., self._velocity_idx+1:]], axis=-1)
        return state

    def step(self, action):
        state, reward, is_terminal, info = self._env.step(action)
        if not self._terminate_on_fall and not ('TimeLimit.truncated' in info and info['TimeLimit.truncated']):
            if is_terminal:
                reward -= self._alive_reward # only set for humanoid so far
            is_terminal = False
        state = self.transform_state(state)
        return state, reward, is_terminal, info

    def reset(self):
        state = self._env.reset()
        state = self.transform_state(state)
        return state

    def render(self, mode='human'):
        self._env.render(mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._env.seed(seed)

    def __str__(self):
        return str(self._env)

    def __enter__(self):
        return self._env.__enter__()

    def __exit__(self, *args):
        return self._env.__exit__(*args)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def np_random(self):
        return self._env.unwrapped.np_random

class NoVelHumanoid(humanoid.HumanoidEnv):
    def __init__(self):
        self._terminate_on_fall = False
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (290,))

    def step(self, action):
        state, reward, is_terminal, info = super().step(action)
        if not self._terminate_on_fall and not ('TimeLimit.truncated' in info and info['TimeLimit.truncated']):
            if is_terminal:
                reward -= 5
            is_terminal = False
        return state, reward, is_terminal, info

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat[2:],
                               data.cinert.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

class NoVelHumanoid2(humanoid.HumanoidEnv):
    def __init__(self):
        self._terminate_on_fall = False
        self._stay_on_fall = False
        self._fallen = False
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (24 + 23 - 4,))

    def reset(self):
        self._fallen = False
        return super().reset()

    def step(self, action):
        if self._fallen and self._stay_on_fall:
            action = np.zeros_like(action)
        state, reward, is_terminal, info = super().step(action)
        if is_terminal and not self._terminate_on_fall:
            self._fallen = True
            reward -= 5
            is_terminal = False
        return state, reward, is_terminal, info

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat[2:]])


if 'NVHalfCheetah-v2' not in registration.registry.env_specs:
    registration.register(
            id='NVHalfCheetah-v2',
            entry_point='paper_experiments.experiments.imitation.nv_environments:NVEnv',
            kwargs={'base_env': 'HalfCheetah-v2', 'velocity_idx': 8,'terminate_on_fall': False},
            max_episode_steps=1000,
            reward_threshold=4800.0
    )
if 'NoComHumanoid-v2' not in registration.registry.env_specs:
    registration.register(
            id='NoComHumanoid-v2',
            entry_point='paper_experiments.experiments.imitation.nv_environments:NoVelHumanoid2',
            kwargs={},
            max_episode_steps=1000
    )
