from typing import Optional, Set, Tuple
import torch
import torch.nn.functional as f
import numpy as np


class MaskLayer(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer('mask', mask)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return _input * self.mask


class SimpleRealNVP(torch.nn.Module):
    def __init__(self, input_dim: int, context_dim: int, hdim: int, num_bijectors: int, random_ordering: bool=False):
        super().__init__()
        self._input_dim = input_dim
        self._context_dim = context_dim

        bijectors = []
        for i in range(num_bijectors):
            input_mask = torch.ones((self._input_dim + self._context_dim,))
            output_mask = torch.ones((self._input_dim + self._context_dim,))
            if i % 2 == 0:
                input_mask[:input_dim//2] = 0
                output_mask[input_dim//2:] = 0

            else:
                input_mask[input_dim // 2:input_dim] = 0
                output_mask[:input_dim//2] = 0
                output_mask[input_dim:] = 0

            if random_ordering:
                if i % 2 == 0:
                    ordering = np.random.permutation(self._input_dim)
                input_mask[:self._input_dim] = input_mask[:self._input_dim][ordering]
                output_mask[:self._input_dim] = output_mask[:self._input_dim][ordering]


            output_mask = output_mask.repeat(2)
            bijectors.append(torch.nn.Sequential(
                MaskLayer(input_mask),
                torch.nn.Linear(input_dim + context_dim, hdim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hdim, hdim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hdim, 2 * (input_dim + context_dim)),
                MaskLayer(output_mask)
            ))
        self._bijectors = torch.nn.ModuleList(bijectors)
        self._normal = torch.distributions.Normal(0, 1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor]=None):
        assert x.shape[1] == self._input_dim and (context is None or context.shape[1] == self._context_dim)
        invlogdetjac: torch.Tensor = 0
        if context is not None:
            x = torch.cat([x, context], dim=1)
        for i, bijector in enumerate(self._bijectors):
            out = bijector(x)
            translate, log_scale = out.chunk(2, dim=1)
            log_scale = f.tanh(log_scale)
            x = (x + translate) * log_scale.exp()
            invlogdetjac = invlogdetjac + log_scale

        x = x[:, :self._input_dim]
        return invlogdetjac[:, :self._input_dim] + self._normal.log_prob(x)

    def generate_samples(self, n: int, context: Optional[torch.Tensor], device: Optional[torch.device]=None):
        if context is not None:
            device = context.device
        x = torch.randn((n, self._input_dim), device=device)
        if context is not None:
            x = torch.cat([x, context], dim=1)
        for j, bijector in enumerate(reversed(self._bijectors)):
            translate, log_scale = bijector(x).chunk(2, dim=1)
            log_scale = f.tanh(log_scale)
            x = x/log_scale.exp() - translate

        return x[:, :self._input_dim]
