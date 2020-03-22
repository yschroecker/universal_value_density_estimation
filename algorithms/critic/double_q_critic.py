from algorithms.critic import temporal_difference
from algorithms.data import TransitionSequence

import torch


class DoubleQModel(torch.nn.Module):
    def __init__(self, q1: torch.nn.Module, q2: torch.nn.Module):
        super().__init__()
        self._q1 = q1
        self._q2 = q2

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self._q1(state, action), self._q2(state, action)


class DoubleQCritic(temporal_difference.TemporalDifferenceBase):
    def __init__(self, q1: torch.nn.Module, q2: torch.nn.Module, target_policy: torch.nn.Module,
                 target_update_step: float, discount_factor: float, action_noise_stddev: float):
        super().__init__(DoubleQModel(q1, q2), 2, target_update_step)
        self._online_q1 = q1
        self._target_policy = target_policy
        self._discount_factor = discount_factor
        self._action_noise_stddev = action_noise_stddev

    def _update(self, batch: TransitionSequence, return_mean: bool=True, *args, **kwargs) -> torch.Tensor:
        states = batch.states[:, 0]
        actions = batch.actions[:, 0]
        rewards = batch.rewards[:, 0]
        bootstrap_weights = self._discount_factor * batch.terminal_weight[:, 0]
        next_states = batch.next_states[:, 0]

        next_actions = self._target_policy(next_states)
        next_actions = next_actions + torch.randn_like(actions) * self._action_noise_stddev

        next_q1, next_q2 = self._target_network(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2).squeeze()

        online_q1, online_q2 = self._online_network(states, actions)

        target_q = (rewards + bootstrap_weights * next_q).detach()

        td_loss_1 = (online_q1.squeeze() - target_q) ** 2
        td_loss_2 = (online_q2.squeeze() - target_q) ** 2
        if return_mean:
            return (td_loss_1 + td_loss_2).mean()
        else:
            return td_loss_1 + td_loss_2

    def q_value(self, state: torch.Tensor, action: torch.Tensor):
        return self._online_q1(state, action)

