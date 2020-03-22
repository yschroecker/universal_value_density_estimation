from typing import NamedTuple, Optional
import torch
import functools


class TDBatch(NamedTuple):
    states: torch.Tensor
    actions: Optional[torch.Tensor]
    intermediate_returns: torch.Tensor
    bootstrap_states: torch.Tensor
    bootstrap_actions: Optional[torch.Tensor]
    bootstrap_weights: torch.Tensor


class TransitionSequence(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    timeout_weight: torch.Tensor
    terminal_weight: torch.Tensor
    action_log_prob: torch.Tensor

