from typing import Callable, Union, NamedTuple, Tuple, Optional
import abc

import numpy as np
import torch
import torch.multiprocessing
import time
from concurrent import futures

from algorithms import environment, data
from algorithms.agents.common import asynchronous_replay_agent
from algorithms.critic import temporal_difference, double_q_critic
from replay import replay_description, ram_buffer
from replay.sampler import offset_sampler
from replay.replay_description import ReplayColumn as RC
from workflow import reporting
import copy
from policies import policy, gaussian
from algorithms.critic import double_q_critic


class OutputTransform(torch.nn.Module):
    def __init__(self, input_module: torch.nn.Module, transform: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self._module = input_module
        self._transform = transform

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._transform(self._module(*args, **kwargs))


class UVDParams(NamedTuple):
    discount_factor: float
    critic_learning_rate: float
    policy_learning_rate: float
    density_learning_rate: float
    target_update_step: float
    target_action_noise: float
    exploration_noise: float

    replay_size: int
    min_replay_size: int
    batch_size: int
    density_replay_size: Optional[int] = None
    density_burnin: int = 0
    sequence_length: int = 4

    burnin: int = 0

    gradient_clip: float = np.inf
    num_envs: int = 1
    actor_device: str = 'cpu'

    her_k: int = 4
    replace_goal_fraction: float = 0.
    validation_fraction: float = 0.0

    shuffle_goals: bool = False
    step_limit: float = 0.


class HerTransitionSequence(NamedTuple, data.TransitionSequence):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    timeout_weight: torch.Tensor
    terminal_weight: torch.Tensor
    action_log_prob: torch.Tensor
    time_left: torch.Tensor
    achieved_goal: torch.Tensor
    target_goal: torch.Tensor


class UVDCritic(temporal_difference.TemporalDifferenceBase):
    def __init__(self, q1: torch.nn.Module, q2: torch.nn.Module, density_q_model, goal_dim: int, target_policy: torch.nn.Module,
                 target_update_step: float, discount_factor: float, action_noise_stddev: float, replace_goal_fraction: float, shuffle_goals: bool,
                 params: UVDParams):
        super().__init__(double_q_critic.DoubleQModel(q1, q2), 2, target_update_step)
        self._online_q1 = q1
        self._density_q_model = density_q_model
        self._target_policy = target_policy
        self._discount_factor = discount_factor
        self._action_noise_stddev = action_noise_stddev
        self._goal_dim = goal_dim
        self._replace_goal_fraction = replace_goal_fraction
        self._largest_target = 0.
        self._shuffle_goals = shuffle_goals
        self._params = params
        reporting.register_field("target_fraction")
        reporting.register_field("target_q")

    def _update(self, batch: data.TransitionSequence, return_mean: bool = True, *args, **kwargs) -> torch.Tensor:
        states = batch.states[:, 0]
        actions = batch.actions[:, 0]
        bootstrap_weights = self._discount_factor * batch.terminal_weight[:, 0]
        next_states = batch.next_states[:, 0]

        goal = states[:, :self._goal_dim]
        if self._shuffle_goals:
            goal = states[torch.randperm(states.shape[0]), :self._goal_dim]
            states = torch.cat([goal, states[:, self._goal_dim:]], dim=1)
            next_states = torch.cat([goal, next_states[:, self._goal_dim:]], dim=1)

        next_actions = self._target_policy(next_states)
        next_actions = next_actions + torch.randn_like(actions) * self._action_noise_stddev

        next_q1, next_q2 = self._target_network(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2).squeeze()

        online_q1, online_q2 = self._online_network(states, actions)

        td_target_q = (bootstrap_weights * next_q).detach()
        density_target_q = self._density_q_model.reward(goal, next_states, next_actions).detach().squeeze()
        target_q = torch.max(td_target_q, density_target_q)
        target_idx = (td_target_q > density_target_q).float().mean()
        reporting.iter_record("target_fraction", target_idx.item())
        reporting.iter_record("target_q", target_q.mean().item())

        td_loss_1 = (online_q1.squeeze() - target_q) ** 2
        td_loss_2 = (online_q2.squeeze() - target_q) ** 2
        if return_mean:
            return (td_loss_1 + td_loss_2).mean()
        else:
            return td_loss_1 + td_loss_2

    def q_value(self, state: torch.Tensor, action: torch.Tensor):
        return self._online_q1(state, action)


class UVDReplayAgent(asynchronous_replay_agent.AsynchronousReplayAgentBase, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        buffer_dim = self._replay_description.num_columns
        device = self._device
        self._sampler = offset_sampler.OffsetSampler(self._buffer, self._params.sequence_length,
                                                     self._replay_description.get_index("weight_column"),
                                                     self._replay_description.get_index("cum_weight_column"),
                                                     self._replay_description.get_index("time_left"),
                                                     self._params.discount_factor, soft_cutoff=False)

        self._density_buffer = ram_buffer.RamBuffer(self._params.density_replay_size, buffer_dim, device)
        self._density_sampler = offset_sampler.OffsetSampler(self._density_buffer, self._params.sequence_length,
                                                             self._replay_description.get_index("weight_column"),
                                                             self._replay_description.get_index("cum_weight_column"),
                                                             self._replay_description.get_index("time_left"),
                                                             self._params.discount_factor, soft_cutoff=False)

        self._valid_buffer = ram_buffer.RamBuffer(int(self._params.replay_size * self._params.validation_fraction),
                                                  buffer_dim, device)
        self._valid_sampler = offset_sampler.OffsetSampler(self._valid_buffer, self._params.sequence_length,
                                                           self._replay_description.get_index("weight_column"),
                                                           self._replay_description.get_index("cum_weight_column"),
                                                           self._replay_description.get_index("time_left"),
                                                           self._params.discount_factor, soft_cutoff=False)

    def get_description(self, env):
        goal_dim = env.goal_dim
        return replay_description.ReplayDescription([
            RC("states", self._state_dim),
            RC("actions", self._action_dim),
            RC("rewards", 0),
            RC("next_states", self._state_dim),
            RC("timeout_weight", 0),
            RC("terminal_weight", 0),
            RC("action_log_prob", 0),
            RC("weight_column", 0, 0),
            RC("cum_weight_column", 0, 0),
            RC("time_left", 0, 0),
            RC("achieved_goal", goal_dim),
            RC("target_goal", goal_dim)
        ], HerTransitionSequence)

    def _add_episode(self, new_trajectory: HerTransitionSequence):
        if np.random.rand() < self._params.validation_fraction:
            self._valid_buffer.add_samples(new_trajectory)
        else:
            self._buffer.add_samples(new_trajectory)
            if self._params.density_replay_size is not None:
                self._density_buffer.add_samples(new_trajectory)


class UVDTD3(UVDReplayAgent):
    def __init__(self, make_env: Callable[[], environment.Environment[Union[np.ndarray, int]]], device: torch.device,
                 goal_r: torch.nn.Module, q1: torch.nn.Module, q2: torch.nn.Module, policy_net: torch.nn.Module,
                 params: UVDParams):
        super().__init__(make_env, device, params)
        spec_env = make_env()
        action_dim = spec_env.action_dim

        self._goal_r = goal_r
        self._online_policy = policy_net
        self._target_policy = copy.deepcopy(policy_net)
        self._behavior_policy = gaussian.SphericalGaussianPolicy(
            device, action_dim, self._target_policy, fixed_noise=self._params.exploration_noise, eval_use_mean=True)
        self._critic = UVDCritic(
                q1, q2, goal_r, spec_env.goal_dim, self._target_policy, params.target_update_step,
                params.discount_factor, params.target_action_noise, params.replace_goal_fraction,
                params.shuffle_goals, params)

        self._target_update_rate = 2
        self._update_count = -1
        self._goal_dim = spec_env.goal_dim

        self._critic_optim = torch.optim.RMSprop(self._critic.parameters, lr=params.critic_learning_rate)
        self._r_optim = torch.optim.RMSprop(self._goal_r.parameters(), lr=params.density_learning_rate,
                                            weight_decay=1e-5)
        self._actor_optim = torch.optim.RMSprop(self._online_policy.parameters(), lr=params.policy_learning_rate)

        reporting.register_field("td_loss")
        reporting.register_field("r_loss")
        reporting.register_field("mean_r")
        reporting.register_field("mean_r_with_goal")
        reporting.register_field("max_r_with_goal")
        reporting.register_field("min_r_with_goal")
        reporting.register_field("actor_loss")
        reporting.register_field("valid_r_loss")
        reporting.register_field("valid_target_r")
        reporting.register_field("valid_target_v")

    def _update_density_estimator(self, offset_batch: HerTransitionSequence, offsets: torch.Tensor):
        r_loss = -(offsets * self._goal_r(
            offset_batch.achieved_goal[:, 1, :],
            offset_batch.states[:, 0, :],
            offset_batch.actions[:, 0, :]).squeeze())

        self._r_optim.zero_grad()
        r_loss.mean().backward()
        self._r_optim.step()

        reporting.iter_record("r_loss", r_loss.mean().item())

    def _update_value(self, batch: HerTransitionSequence):
        if self._update_count >= self._params.density_burnin:
            td_loss = self._critic.update_loss(batch)
            self._critic_optim.zero_grad()
            td_loss.backward()
            if np.isfinite(self._params.gradient_clip):
                torch.nn.utils.clip_grad_norm_(self._critic.parameters, self._params.gradient_clip, norm_type=2)
            self._critic_optim.step()
            reporting.iter_record("td_loss", td_loss.item())

    def _update_policy(self, batch: HerTransitionSequence):
        # update policy
        if self._update_count >= self._params.burnin:
            actor_loss = -self._critic.q_value(batch.states[:, 0], self._online_policy(batch.states[:, 0])).mean()

            self._actor_optim.zero_grad()
            actor_loss.backward()
            self._actor_optim.step()

            if self._update_count % self._target_update_rate == 0:
                for online_param, target_param in zip(self._online_policy.parameters(), self._target_policy.parameters()):
                    target_param.requires_grad = False
                    target_param.data = ((1 - self._params.target_update_step) * target_param +
                                         self._params.target_update_step * online_param)

            reporting.iter_record("actor_loss", actor_loss.item())

    def _update(self):
        self._update_count += 1

        # Report validation losses
        if self._update_count % 100 == 0 and self._valid_buffer.size > 0:
            valid_batch, _ = self._density_sampler.sample_discounted_offset(self._params.batch_size)
            valid_batch = self._replay_description.parse_sample(valid_batch)

            valid_loss = -(self._goal_r(
                valid_batch.achieved_goal[:, 1, :],
                valid_batch.states[:, 0, :],
                valid_batch.actions[:, 0, :])).mean()
            reporting.iter_record("valid_r_loss", valid_loss.item())

            valid_target_r = self._goal_r.reward(
                valid_batch.target_goal[:, 0, :],
                valid_batch.states[:, 0, :],
                valid_batch.actions[:, 0, :]).mean()
            valid_target_v = self._critic.q_value(
                valid_batch.states[:, 0, :],
                valid_batch.actions[:, 0, :]).mean()
            reporting.iter_record("valid_target_r", valid_target_r.item())
            reporting.iter_record("valid_target_v", valid_target_v.item())

        batch = self._replay_description.parse_sample(self._buffer.sample_sequence(self._params.batch_size, 2))
        offset_batch, density_target_offset = self._density_sampler.sample_uniform_offset(self._params.batch_size)
        offset_batch = self._replay_description.parse_sample(offset_batch)
        offset_weights = self._params.discount_factor ** (offset_batch.states.new(density_target_offset) - 1)

        self._update_density_estimator(offset_batch, offset_weights)
        self._update_value(batch)
        self._update_policy(batch)

    def freeze_policy(self, device: torch.device) -> policy.Policy:
        return self._behavior_policy.clone()

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self._behavior_policy.mode(torch.from_numpy(state.astype(np.float32)).to(self._behavior_policy.device))

    def sample_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        return self._behavior_policy.sample(state, return_logprob=True)
