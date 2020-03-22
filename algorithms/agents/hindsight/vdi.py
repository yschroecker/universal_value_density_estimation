from typing import Callable, Union, NamedTuple, Tuple, Optional, List
import abc

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as f

from generative import rnvp
from algorithms import environment, data
from algorithms.agents.common import asynchronous_replay_agent
from algorithms.critic import double_q_critic, temporal_difference
from replay import replay_description, ram_buffer
from replay.sampler import offset_sampler
from replay.replay_description import ReplayColumn as RC
from workflow import reporting
import copy
from policies import policy, gaussian
import util


class VDIParams(NamedTuple):
    discount_factor: float
    policy_learning_rate: float
    density_learning_rate: float
    critic_learning_rate: float
    target_update_step: float
    exploration_noise: float
    replay_size: int
    min_replay_size: int
    batch_size: int
    density_replay_size: int = 50000
    density_update_rate: int = 50000
    density_update_rate_burnin: int = 0
    burnin_density_update_rate: int = 5000
    density_burnin: int = 4000
    action_noise_stddev: float = 0.
    temporal_smoothing: float = 0.
    step_limit: float = 0.
    lr_decay_rate: float = 0.33
    lr_decay_iterations: List[int] = [] # [250000, 1000000]
    exploration_decay: List[Tuple[int, float]] = [] 

    sequence_length: int = 4
    density_factor: float = 1.0
    gradient_clip: float = np.inf
    critic_gradient_clip: float = np.inf
    burnin: int = 9000
    num_envs: int = 1
    actor_device: str = 'cpu'
    density_l2: float = 1e-5
    policy_l2: float = 0
    critic_l2: float = 0
    spatial_smoothing: float = 0.1


class DoubleQModel(torch.nn.Module):
    def __init__(self, q1: torch.nn.Module, q2: torch.nn.Module):
        super().__init__()
        self._q1 = q1
        self._q2 = q2

    def forward(self, state: torch.Tensor, action: torch.Tensor, target_state: torch.Tensor):
        return self._q1(state, action, target_state), self._q2(state, action, target_state)


class VDICritic(temporal_difference.TemporalDifferenceBase):
    def __init__(self, q1: torch.nn.Module, q2: torch.nn.Module, density_q_model, state_density_model, target_policy: torch.nn.Module,
                 params: VDIParams):
        super().__init__(DoubleQModel(q1, q2), 2, params.target_update_step)
        self._online_q1 = q1
        self._density_q_model = density_q_model
        self._state_density_model = state_density_model
        self._target_policy = target_policy
        self._discount_factor = params.discount_factor
        self._action_noise_stddev = params.action_noise_stddev
        self._largest_target = 0.
        self._params = params
        self._temporal_smoothing = params.temporal_smoothing
        reporting.register_field("target_fraction")
        reporting.register_field("target_q")
        reporting.register_field("max_state_density")
        reporting.register_field("min_state_density")
        reporting.register_field("mean_state_density")
        reporting.register_field("state_density_bound")
        reporting.register_field("terminal_weight")

    def _update(self, batch: data.TransitionSequence, target_states: torch.Tensor, *args, weights: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        states = batch.states
        actions = batch.actions
        bootstrap_weights = self._discount_factor * batch.terminal_weight
        next_states = batch.next_states

        next_actions = self._target_policy(next_states)
        next_actions = next_actions + torch.randn_like(actions) * self._action_noise_stddev

        next_q1, next_q2 = self._target_network(next_states, next_actions, target_states)
        next_q = torch.min(next_q1, next_q2).squeeze()

        online_q1, online_q2 = self._online_network(states, actions, target_states)

        td_target_q = (bootstrap_weights * next_q).detach()

        density_target_q = (self._density_q_model(state=states, action=actions, target_state=target_states)).exp()
        if self._temporal_smoothing > 0.:
            density_target_q = self._temporal_smoothing * td_target_q + (1 - self._temporal_smoothing) * density_target_q


        target_q = batch.terminal_weight * torch.max(td_target_q, density_target_q)
        target_idx = (td_target_q > density_target_q).float().mean()
        reporting.iter_record("terminal_weight", batch.terminal_weight.sum().item())
        reporting.iter_record("target_q", target_q.mean().item())
        reporting.iter_record("target_fraction", target_idx.item())


        td_loss_1 = f.smooth_l1_loss(online_q1.squeeze(), target_q)
        td_loss_2 = f.smooth_l1_loss(online_q2.squeeze(), target_q)
        return (td_loss_1 + td_loss_2).mean()

    def q_value(self, state: torch.Tensor, action: torch.Tensor, target_state: torch.Tensor):
        return self._online_q1(state, action, target_state)


class VDIReplayAgent(asynchronous_replay_agent.AsynchronousReplayAgentBase, abc.ABC):
    def __init__(self, state_dim, *args, **kwargs):
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
                                                             self._replay_description.get_index("density_weight_column"),
                                                             self._replay_description.get_index("density_cum_weight_column"),
                                                             self._replay_description.get_index("density_time_left"),
                                                             self._params.discount_factor**state_dim, soft_cutoff=False)

    def get_description(self, env):
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
            RC("density_weight_column", 0, 0),
            RC("density_cum_weight_column", 0, 0),
            RC("density_time_left", 0, 0),
        ], data.TransitionSequence)

    def _add_episode(self, new_trajectory: data.TransitionSequence):
        self._buffer.add_samples(new_trajectory)
        self._density_buffer.add_samples(new_trajectory)


class VDI(VDIReplayAgent):
    def __init__(self, make_env: Callable[[], environment.Environment[Union[np.ndarray, int]]], device: torch.device,
                 density_model: rnvp.SimpleRealNVP, state_density_model: rnvp.SimpleRealNVP,
                 policy_net: torch.nn.Module, q1: torch.nn.Module, q2: torch.nn.Module, params: VDIParams,
                 demo_states: torch.Tensor, demo_actions: Optional[torch.Tensor]):
        spec_env = make_env()
        state_dim = spec_env.state_dim
        action_dim = spec_env.action_dim
        super().__init__(state_dim, make_env, device, params)
        print(demo_states.shape)

        self._density_update_rate = params.burnin_density_update_rate
        self._online_q1 = q1
        self._density_model = density_model
        self._target_density_model = util.target_network(density_model)
        self._state_density_model = state_density_model
        self._target_state_density_model = util.target_network(state_density_model)
        self._online_policy = policy_net
        self._target_policy = util.target_network(policy_net)
        self._behavior_policy = gaussian.SphericalGaussianPolicy(
            device, action_dim, self._target_policy, fixed_noise=self._params.exploration_noise, eval_use_mean=True)

        self.spatial_smoothing = self._params.spatial_smoothing
        self._target_update_rate = 1
        self._update_count = -1
        self._demo_weights = np.ones((demo_states.shape[0], ))/demo_states.shape[0]

        self._density_optim = torch.optim.RMSprop(
            set(self._density_model.parameters()) | set(self._state_density_model.parameters()),
            lr=params.density_learning_rate, weight_decay=self._params.density_l2)
        self._critic = VDICritic(q1, q2, self._target_density_model, self._target_state_density_model, self._target_policy, params)
        self._actor_optim = torch.optim.RMSprop(self._online_policy.parameters(), lr=params.policy_learning_rate,
                                                weight_decay=self._params.policy_l2)
        self._critic_optim = torch.optim.RMSprop(self._critic.parameters, lr=params.critic_learning_rate,
                                                 weight_decay=params.critic_l2)
        if len(params.lr_decay_iterations) > 0:
            self._actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._actor_optim, params.lr_decay_iterations,
                                                                        params.lr_decay_rate)
            self._critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._critic_optim, params.lr_decay_iterations,
                                                                        params.lr_decay_rate)

        self._demo_states = demo_states
        self._demo_actions = demo_actions
        self._last_imagined_samples = None

        reporting.register_field("q_norm")
        reporting.register_field("policy_norm")
        reporting.register_field("density_loss")
        reporting.register_field("state_density_loss")
        reporting.register_field("valid_density_loss")
        reporting.register_field("actor_loss")
        reporting.register_field("bc_loss")
        reporting.register_field("td_loss")
        reporting.register_field("actor_lr")
        reporting.register_field("critic_lr")

    def _update_density_estimator(self, offset_batch: data.TransitionSequence, offset_weights: torch.Tensor):
        target_states = offset_batch.next_states[:, 1, :] + torch.randn_like(offset_batch.next_states[:, 1, :]) * self.spatial_smoothing
        density_loss = -(self._density_model(
            state=offset_batch.states[:, 0, :],
            action=offset_batch.actions[:, 0, :],
            target_state=target_states)).squeeze().mean()
        state_density_loss = -self._state_density_model(target_states).mean()

        reporting.iter_record("density_loss", density_loss.item())
        self._density_optim.zero_grad()
        (density_loss + state_density_loss).backward()
        self._density_optim.step()

    def _update_policy(self, offset_batch):
        # update policy
        if self._update_count >= self._params.burnin:
            sample_batch_indices = np.searchsorted(self._state_density_cum_weights, np.random.rand(self._params.batch_size))
            demo_states = self._demo_states[sample_batch_indices]

            batch_states = offset_batch.states

            policy_actions = self._online_policy(batch_states)
            target_density = self._critic.q_value(batch_states, policy_actions, demo_states)

            actor_loss = -target_density.mean()

            self._actor_optim.zero_grad()
            actor_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self._online_policy.parameters(), self._params.gradient_clip, 2)
            reporting.iter_record("policy_norm", total_norm)
            self._actor_optim.step()

            if self._update_count % self._target_update_rate == 0:
                for online_param, target_param in zip(self._online_policy.parameters(), self._target_policy.parameters()):
                    target_param.requires_grad = False
                    target_param.data = ((1 - self._params.target_update_step) * target_param +
                                         self._params.target_update_step * online_param)

            reporting.iter_record("actor_loss", actor_loss.item())
            if len(self._params.exploration_decay) > 0:
                self._behavior_policy.stepwise_exploration_decay(self._params.exploration_decay, self._update_count)

    def _update_value(self, batch: data.TransitionSequence):
        if self._update_count >= self._params.density_burnin:
            sample_batch_indices = np.random.choice(self._demo_states.shape[0], size=self._params.batch_size, replace=True)
            demo_states = self._demo_states[sample_batch_indices]

            target_states = demo_states
            td_loss = self._critic.update_loss(batch, target_states, weights=None)
            self._critic_optim.zero_grad()
            td_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self._critic.parameters, self._params.critic_gradient_clip, norm_type=2)
            reporting.iter_record("q_norm", total_norm)
            self._critic_optim.step()
            reporting.iter_record("td_loss", td_loss.item())

    def _update(self):
        self._update_count += 1
        if len(self._params.lr_decay_iterations) > 0:
            self._actor_scheduler.step()
            self._critic_scheduler.step()
            reporting.iter_record("actor_lr", self._actor_scheduler.get_lr()[0])
            reporting.iter_record("critic_lr", self._actor_scheduler.get_lr()[0])
        if self._update_count >= self._params.density_update_rate_burnin:
            self._density_update_rate = self._params.density_update_rate


        if self._update_count % self._density_update_rate == 0:
            self._density_optim.zero_grad()
            self._target_state_density_model.load_state_dict(copy.deepcopy(self._state_density_model.state_dict()))
            self._target_density_model.load_state_dict(copy.deepcopy(self._density_model.state_dict()))

            state_density = self._target_state_density_model(self._demo_states).exp()
            reporting.iter_record("max_state_density", state_density.max().item())
            reporting.iter_record("min_state_density", state_density.min().item())
            reporting.iter_record("mean_state_density", state_density.mean().item())
            reporting.iter_record("state_density_bound", state_density.mean().item() / self._params.density_factor)
            state_density_weights = state_density/state_density.mean()
            state_density_weights = torch.clamp(1/state_density_weights, max=self._params.density_factor)
            self._state_density_cum_weights = torch.cumsum(state_density_weights.squeeze(), 0).cpu().detach().numpy()
            self._state_density_cum_weights /= self._state_density_cum_weights[-1]

        offset_batch, density_target_offset = self._density_sampler.sample_discounted_offset(self._params.batch_size)
        offset_batch = self._replay_description.parse_sample(offset_batch)
        offset_weights = self._params.discount_factor ** (offset_batch.states.new(density_target_offset) - 1)

        self._update_density_estimator(offset_batch, offset_weights)


        single_batch = self._buffer.sample(self._params.batch_size)
        single_batch = self._replay_description.parse_sample(single_batch)
        self._update_value(single_batch)

        self._update_policy(single_batch)


    def freeze_policy(self, device: torch.device) -> policy.Policy:
        return self._behavior_policy.clone()

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self._behavior_policy.mode(torch.from_numpy(state.astype(np.float32)).to(self._behavior_policy.device))

    def sample_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        return self._behavior_policy.sample(state, return_logprob=True)
