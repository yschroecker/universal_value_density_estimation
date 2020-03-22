from typing import NamedTuple, Callable, Tuple
from algorithms import environment
from algorithms import data
from algorithms.critic import vtrace
from algorithms.agents import v_traced_a2c
from algorithms.agents.common import asynchronous_replay_agent
from policies import policy
from workflow import reporting

import torch
import numpy as np


class IMPALAParams(asynchronous_replay_agent.AsyncReplayAgentParams, v_traced_a2c.VTracedA2CParams, NamedTuple):
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

    num_envs: int = 1
    actor_device: str = 'cpu'

    buffer_type: str = 'ram'


class IMPALA(v_traced_a2c.VTracedA2CMixin, asynchronous_replay_agent.AsynchronousReplayAgent):
    def freeze_policy(self, device: torch.device) -> policy.Policy:
        cloned_policy = self._policy.clone()
        cloned_policy.to(device)
        return cloned_policy
