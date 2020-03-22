from typing import Any

import torch

from algorithms.critic import temporal_difference
from policies import policy
from algorithms import data


class VTrace(temporal_difference.ValueTemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, policy: policy.Policy, target_update_rate: int, discount_factor: float,
                 iw_trunc_1: float=1., iw_trunc_2: float=1.):
        super().__init__(model, target_update_rate)
        self._discount_factor = discount_factor
        self._iw_trunc_1 = iw_trunc_1
        self._iw_trunc_2 = iw_trunc_2
        self._policy = policy
        self.advantage_targets = None
        self.importance_weights = None

    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self._online_network(states).squeeze()

    def _update(self, batch: data.TransitionSequence) -> torch.Tensor:
        """
        :param batch: has tensors of the form BxTxX where X is the dimensionality of the item (e.g. state_dim)
        """
        values_o = self._online_network(batch.states[:, 0]).squeeze()

        batch_size = batch.states.shape[0]
        time_steps = batch.states.shape[1]
        state_values = self._target_network(
            batch.states.reshape((batch_size * time_steps, -1))).reshape((batch_size, time_steps))
        next_state_values = self._target_network(
            batch.next_states.reshape((batch_size * time_steps, -1))).reshape((batch_size, time_steps))
        target_action_prob = self._policy.log_probability(
            batch.states.reshape((batch_size * time_steps, -1)),
            batch.actions.reshape((batch_size * time_steps, -1))
        ).reshape(batch_size, time_steps)
        importance_weights = torch.clamp((target_action_prob - batch.action_log_prob).exp(), max=self._iw_trunc_1).detach()
        target_values = state_values[:, 0]
        prev_iw_prod = target_values.new_ones((batch_size,))  # product of twice truncated importance weights

        advantage_target_values = next_state_values[:, 0]
        advantage_prev_iw_prod = target_values.new_ones((batch_size,))  # product of twice truncated importance weights
        timeout_weight = state_values.new_ones((batch_size, time_steps))  # make persistent?
        for t in range(time_steps):
            if t != time_steps - 1:
                timeout_weight[:, t + 1] = timeout_weight[:, t] * batch.timeout_weight[:, t]

            td_error = importance_weights[:, t] * (
                    batch.rewards[:, t] + self._discount_factor * batch.terminal_weight[:, t] * next_state_values[:, t] -
                    state_values[:, t])
            target_values = (target_values +
                             timeout_weight[:, t] * self._discount_factor**t * prev_iw_prod * td_error)

            prev_iw_prod = prev_iw_prod * torch.clamp(importance_weights[:, t], max=self._iw_trunc_2)
            if t >= 1:
                advantage_target_values = (advantage_target_values +
                                           timeout_weight[:, t] * self._discount_factor**(t - 1) * advantage_prev_iw_prod * td_error)
                advantage_prev_iw_prod = advantage_prev_iw_prod * torch.clamp(importance_weights[:, t], max=self._iw_trunc_2)

        self.advantage_targets = advantage_target_values
        self.importance_weights = importance_weights[:, 0]

        return torch.mean((values_o.squeeze() - target_values.detach()) ** 2)

