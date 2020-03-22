from typing import NamedTuple, Callable, Tuple
from algorithms import environment
from algorithms import data
from algorithms.critic import vtrace
from algorithms.agents.common import synchronous_replay_agent
from policies import policy
from workflow import reporting

import torch
import numpy as np


class VTracedA2CParams(synchronous_replay_agent.ReplayAgentParams, NamedTuple):
    discount_factor: float
    learning_rate: float
    batch_size: int

    sequence_length: int
    replay_size: int
    min_replay_size: int
    gradient_clip: float = np.inf
    value_target_update_rate: int = 1
    entropy_regularization: float = 0.
    burnin: int = 0
    policy_update_rate: int = 1

    buffer_type: str = 'ram'
    steps_per_update: int = 1


class VTracedA2CMixin:
    def __init__(self,
                 make_env: Callable[[], environment.Environment],
                 policy_: policy.Policy,
                 value_function: torch.nn.Module,
                 params: VTracedA2CParams):
        super().__init__(make_env, list(value_function.parameters())[0].device, params)
        self._policy = policy_
        self._params = params

        self._critic = vtrace.VTrace(
            model=value_function,
            policy=self._policy,
            discount_factor=self._params.discount_factor,
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

    def _update(self, td_batch: data.TransitionSequence):
        td_loss = self._critic.update_loss(td_batch)
        self._value_optimizer.zero_grad()
        td_loss.backward()
        if np.isfinite(self._params.gradient_clip):
            torch.nn.utils.clip_grad_norm_(self._critic.parameters, self._params.gradient_clip, norm_type=2)
        self._value_optimizer.step()

        reporting.iter_record("td_loss",  td_loss.item())
        self._num_updates += 1
        if self._num_updates < self._params.burnin or self._num_updates % self._params.policy_update_rate != 0:
            return

        vtrace_targets = self._critic.advantage_targets.detach()
        importance_weights = self._critic.importance_weights

        advantages = (td_batch.rewards[:, 0] +
                      self._params.discount_factor * vtrace_targets -
                      self._critic.values(td_batch.states[:, 0]))
        advantage_loss = (-importance_weights * self._policy.log_probability(td_batch.states[:, 0], td_batch.actions[:, 0]).squeeze() * advantages.detach()).mean()
        if self._params.entropy_regularization > 0:
            entropy_loss = -self._params.entropy_regularization * self._policy.entropy(td_batch.states[:, 0]).mean()
        if self._params.entropy_regularization > 0:
            loss = advantage_loss + entropy_loss
        else:
            loss = advantage_loss

        reporting.iter_record("advantage_loss", advantage_loss.item())
        if self._params.entropy_regularization > 0:
            reporting.iter_record("entropy_loss", entropy_loss.item())

        self._actor_optimizer.zero_grad()
        loss.backward()
        if np.isfinite(self._params.gradient_clip):
            torch.nn.utils.clip_grad_norm_(self._policy.parameters, self._params.gradient_clip, norm_type=2)
        self._actor_optimizer.step()

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self._policy.mode(torch.from_numpy(state.astype(np.float32)).to(self._policy.device))

    def sample_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        return self._policy.sample(state, return_logprob=True)


class VTracedA2C(VTracedA2CMixin, synchronous_replay_agent.SynchronousReplayAgent):
    pass
