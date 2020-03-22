import torch

from algorithms.critic import temporal_difference
from algorithms import data
from workflow import reporting


class ValueTD(temporal_difference.ValueTemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, target_update_rate: int, *args, **kwargs):
        super().__init__(model, target_update_rate, *args, **kwargs)
        reporting.register_field("td_target_value")

    def _update(self, batch: data.TDBatch, return_mean: bool=True) -> torch.Tensor:
        values_o = self._online_network(batch.states).squeeze()
        next_values_t = self._target_network(batch.bootstrap_states).squeeze()

        target_values = batch.intermediate_returns + batch.bootstrap_weights * next_values_t
        reporting.iter_record("td_target_value", target_values.mean().item())

        if return_mean:
            return torch.mean((values_o.squeeze() - target_values) ** 2)
        else:
            return (values_o.squeeze() - target_values) ** 2

    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self._online_network(states).squeeze()


class QValueTD(temporal_difference.TemporalDifferenceBase):
    def __init__(self, model: torch.nn.Module, target_update_rate: int, *args, **kwargs):
        super().__init__(model, target_update_rate, *args, **kwargs)
        reporting.register_field("td_target_value")

    def _update(self, batch: data.TDBatch) -> torch.Tensor:
        q_values_o = self._online_network(batch.states)
        values_o = q_values_o.gather(dim=1, index=batch.actions.unsqueeze(1))
        next_q_values_t = self._target_network(batch.bootstrap_states)
        next_values_t = next_q_values_t.gather(dim=1, index=batch.bootstrap_actions.unsqueeze(1)).squeeze()

        target_values = batch.intermediate_returns + batch.bootstrap_weights * next_values_t
        reporting.iter_record("td_target_value", target_values.mean().item())
        return torch.mean((values_o.squeeze() - target_values) ** 2)

    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self._online_network(states).squeeze()
