import torch
import torch.nn.functional as f
import copy


def soft_clamp(tensor: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(tensor) * (max_value - min_value) + min_value


def soft_clamp_min(tensor: torch.Tensor, min_value: float) -> torch.Tensor:
    return f.softplus(tensor - min_value) + min_value
    # return torch.sigmoid(tensor) * (max_value - min_value) + min_value


def soft_clamp_max(tensor: torch.Tensor, max_value: float) -> torch.Tensor:
    return -soft_clamp_min(-tensor, -max_value)


def softplus_clamp(tensor: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return soft_clamp_max(soft_clamp_min(tensor, min_value=min_value), max_value=max_value)
    # return -soft_min(-tensor, -max_value)

def target_network(net: torch.nn.Module) -> torch.nn.Module:
    target_net = copy.deepcopy(net)
    for param in target_net.parameters():
        param.requires_grad = False
    target_net = target_net.to(next(net.parameters()).device)
    return target_net

