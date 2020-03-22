from typing import NamedTuple, Callable
from algorithms import environment
from algorithms import data
from algorithms.critic import value_td
from algorithms.agents.common import online_agent
from policies import policy
from workflow import reporting

import torch
import numpy as np


class A2CParams(online_agent.OnlineAgentParams, NamedTuple):
    discount_factor: float
    learning_rate: float
    batch_size: int
    gradient_clip: float = np.inf
    value_target_update_rate: int = 1
    entropy_regularization: float = 0.
    burnin: int = 0

    num_envs: int = 1
    num_steps: int = 1


class A2C(online_agent.OnlineAgent):
    def __init__(self,
                 make_env: Callable[[], environment.Environment],
                 policy_: policy.Policy,
                 value_function: torch.nn.Module,
                 params: A2CParams):
        super().__init__(make_env, list(value_function.parameters())[0].device, params)
        self._policy = policy_
        self._params = params

        self._critic = value_td.ValueTD(
            model=value_function,
            target_update_rate=params.value_target_update_rate
        )
        self._actor_optimizer = torch.optim.RMSprop(
            set(self._policy.parameters),
            eps=0.1,
            lr=params.learning_rate)
        self._value_optimizer = torch.optim.RMSprop(
            value_function.parameters(),
            eps=0.1,
            lr=params.learning_rate)
        reporting.register_field("entropy_loss")
        reporting.register_field("advantage_loss")
        reporting.register_field("td_loss")

        self._num_updates = 0

    def _update(self, td_batch: data.TDBatch):
        td_loss = self._critic.update_loss(td_batch)
        self._value_optimizer.zero_grad()
        td_loss.backward()
        if np.isfinite(self._params.gradient_clip):
            torch.nn.utils.clip_grad_norm_(self._critic.parameters, self._params.gradient_clip, norm_type='inf')
        self._value_optimizer.step()

        reporting.iter_record("td_loss",  td_loss.item())
        self._num_updates += 1
        if self._num_updates < self._params.burnin:
            return

        advantages = (td_batch.intermediate_returns +
                      td_batch.bootstrap_weights * self._critic.values(td_batch.bootstrap_states) -
                      self._critic.values(td_batch.states))
        advantage_loss = (-self._policy.log_probability(td_batch.states, td_batch.actions).squeeze() * advantages.detach()).mean()
        if self._params.entropy_regularization > 0:
            entropy_loss = -self._params.entropy_regularization * self._policy.entropy(td_batch.states).mean()
        if self._params.entropy_regularization > 0:
            loss = advantage_loss + entropy_loss
        else:
            loss = advantage_loss

        reporting.iter_record("advantage_loss", advantage_loss.item())
        if self._params.entropy_regularization > 0:
            reporting.iter_record("entropy_loss", entropy_loss.item())

        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self._policy.mode(torch.from_numpy(state.astype(np.float32)).to(self._policy.device))

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        return self._policy.sample(state)
