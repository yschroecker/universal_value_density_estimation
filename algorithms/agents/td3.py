from typing import NamedTuple, Callable, Union, Tuple

import torch
import copy
import numpy as np

from policies import policy, gaussian
from algorithms.agents.common import asynchronous_replay_agent
from algorithms.critic import double_q_critic
from algorithms import environment
from workflow import reporting


class TD3Params(asynchronous_replay_agent.AsyncReplayAgentParams, NamedTuple):
    discount_factor: float
    critic_learning_rate: float
    policy_learning_rate: float
    target_update_step: float
    target_action_noise: float
    exploration_noise: float

    replay_size: int
    min_replay_size: int
    batch_size: int
    sequence_length: int = 1  # has to be 1

    burnin: int = 0

    gradient_clip: float = np.inf
    buffer_type: str = 'ram'
    steps_per_update: int = 1
    num_envs: int = 1
    actor_device: str = 'cpu'
    step_limit: float = 0.


class TD3(asynchronous_replay_agent.AsynchronousReplayAgent):
    def __init__(self, make_env: Callable[[], environment.Environment[Union[np.ndarray, int]]], device: torch.device,
                 q1: torch.nn.Module, q2: torch.nn.Module, policy_net: torch.nn.Module, params: TD3Params):
        super().__init__(make_env, device, params)
        spec_env = make_env()
        action_dim = spec_env.action_dim

        self._online_policy = policy_net
        self._target_policy = copy.deepcopy(policy_net)
        self._behavior_policy = gaussian.SphericalGaussianPolicy(
            device, action_dim, self._target_policy, fixed_noise=self._params.exploration_noise, eval_use_mean=True)
        self._critic = double_q_critic.DoubleQCritic(q1, q2, self._target_policy, params.target_update_step,
                                                     params.discount_factor, params.target_action_noise)

        self._target_update_rate = 2
        self._update_count = -1

        self._critic_optim = torch.optim.RMSprop(self._critic.parameters, lr=params.critic_learning_rate)
        self._actor_optim = torch.optim.RMSprop(self._online_policy.parameters(), lr=params.policy_learning_rate)

        reporting.register_field("advantage_loss")
        reporting.register_field("td_loss")

    def _update(self):
        batch = self._replay_description.parse_sample(self._buffer.sample(self._params.batch_size)[:, None])
        self._update_count += 1
        td_loss = self._critic.update_loss(batch)

        self._critic_optim.zero_grad()
        td_loss.backward()
        if np.isfinite(self._params.gradient_clip):
            torch.nn.utils.clip_grad_norm_(self._critic.parameters, self._params.gradient_clip, norm_type=2)
        self._critic_optim.step()
        reporting.iter_record("td_loss", td_loss.item())

        if self._update_count >= self._params.burnin:
            actor_loss = -self._critic.q_value(batch.states[:, 0], self._online_policy(batch.states[:, 0])).mean()

            self._actor_optim.zero_grad()
            actor_loss.backward()
            self._actor_optim.step()

            reporting.iter_record("actor_loss", actor_loss.item())

            if self._update_count % self._target_update_rate == 0:
                for online_param, target_param in zip(self._online_policy.parameters(), self._target_policy.parameters()):
                    target_param.requires_grad = False
                    target_param.data = (1 - self._params.target_update_step) * target_param + self._params.target_update_step * online_param

    def freeze_policy(self, device: torch.device) -> policy.Policy:
        return self._behavior_policy.clone()

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self._behavior_policy.mode(torch.from_numpy(state.astype(np.float32)).to(self._behavior_policy.device))

    def sample_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        return self._behavior_policy.sample(state, return_logprob=True)

