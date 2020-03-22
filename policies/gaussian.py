from typing import Sequence, Optional, Tuple, Type

import copy
import torch
import numpy as np

import policies.policy


class SphericalGaussianPolicy(policies.policy.Policy[np.ndarray]):
    def __init__(self, device: torch.device, action_dim: int, network: torch.nn.Module,
                 fixed_noise: Optional[np.ndarray]=None, min_stddev: float=0, eval_use_mean: bool=False):
        self._network = network
        if fixed_noise is None:
            self._fixed_noise = None
        elif type(fixed_noise) is float or type(fixed_noise) is int:
            self._fixed_noise = (torch.ones(action_dim) * fixed_noise).to(device)
        else:
            self._fixed_noise = torch.from_numpy(fixed_noise).to(device)
        self._action_dim = action_dim
        self.min_stddev = min_stddev
        self._noise = None
        self.eval_use_mean = eval_use_mean
        self._device = device
        self._stepwise_decay_step = 0

    def log_linear_exploration_decay(self, min_noise: float, max_noise: float, decay_start: int, decay_end: int, step: int):
        if step <= decay_start:
            self._fixed_noise[:] = max_noise
        elif step >= decay_end:
            self._fixed_noise[:] = min_noise
        else:
            self._fixed_noise[:] = min_noise + (step - decay_start)/(decay_end - decay_start) * (max_noise - min_noise)

    def stepwise_exploration_decay(self, decay_steps: Sequence[Tuple[int, float]], step: int):
        if self._stepwise_decay_step < len(decay_steps) and step > decay_steps[self._stepwise_decay_step][0]:
            self._fixed_noise[:] = decay_steps[self._stepwise_decay_step][1]
            self._stepwise_decay_step += 1

    def clone(self):
        clone_ = copy.deepcopy(self)
        clone_._network.to(self._device)
        return clone_

    def to(self, device: torch.device):
        self._network.to(device)
        self._device = device

    def share_memory(self):
        self._network.share_memory()

    @property
    def device(self) -> torch.device:
        return self._device

    def _gaussian_log_prob(self, actions: torch.Tensor, means: torch.Tensor, logstddevs: torch.Tensor):
        lognorm_constants = -0.5 * (self._action_dim*np.log(2*np.pi).astype(np.float32)).item() - \
            logstddevs.sum(dim=1)
        log_pdf = -0.5 * (((actions - means)/torch.exp(logstddevs))**2).sum(dim=1) + lognorm_constants
        return log_pdf

    def log_probability(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        means, logstddevs = self.statistics(states)
        return self._gaussian_log_prob(actions, means, logstddevs)

    @property
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return self._network.parameters()

    def entropy(self, states: torch.Tensor) -> torch.Tensor:
        mean, logstddevs = self.statistics(states)
        return (0.5 + np.log(np.sqrt(2*np.pi)).astype(np.float32)).item()*self._action_dim + logstddevs.sum(dim=1)

    @property
    def _module(self) -> torch.nn.Module:
        return self._network

    def statistics(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self._network(states)
        if self._fixed_noise is not None:
            means = out
            logstddevs = self._fixed_noise.expand_as(means)
        else:
            means = out[:, :self._action_dim]
            logstddevs = out[:, self._action_dim:]
        return means, logstddevs

    @property
    def action_type(self) -> Type[np.dtype]:
        return np.float32

    def sample_tensor(self, state: torch.Tensor) -> torch.Tensor:
        mean_var, logstddev_var = self.statistics(state)
        action_var = torch.randn_like(mean_var) * logstddev_var.exp() + mean_var
        return action_var

    def sample_from_var(self, state: torch.Tensor, t: int=0, return_logprob: bool=False) -> np.ndarray:
        mean_var, logstddev_var = self.statistics(state)
        action_var = torch.randn_like(mean_var) * logstddev_var.exp() + mean_var
        action = action_var.detach().cpu().numpy().squeeze()
        #mean = mean_var.detach().cpu().numpy().squeeze()
        #logstddev = logstddev_var.detach().cpu().numpy().squeeze()
        #stddev = np.exp(logstddev) + self.min_stddev
        #action_2 = np.random.normal(mean, stddev)

        if isinstance(action, np.ndarray):
            action = np.atleast_1d(action.astype(np.float32))
        else:
            action = np.array([action], dtype=np.float32)
        if return_logprob:
            return action, self._gaussian_log_prob(action_var, mean_var, logstddev_var).item()
        else:
            return action

    def mode(self, state: torch.Tensor) -> np.ndarray:
        mean_var, _ = self.statistics(state)
        return mean_var.cpu().detach().numpy()

