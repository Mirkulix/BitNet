"""Dynamic network expansion utilities."""

from dataclasses import dataclass
from typing import List

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


@dataclass
class ExpansionEvent:
    layer_name: str
    new_neurons: int


class DynamicExpansion:
    """Simple dynamic expansion handler."""

    def __init__(self, model: nn.Module):
        if torch is None:
            raise ImportError("PyTorch required for DynamicExpansion")
        self.model = model
        self.history: List[ExpansionEvent] = []

    def expand_linear(self, layer: nn.Linear, new_out: int):
        weight = layer.weight.data
        new_weight = torch.randn(new_out, layer.in_features, device=weight.device)
        layer.weight = nn.Parameter(torch.cat([weight, new_weight], dim=0))
        if layer.bias is not None:
            bias = layer.bias.data
            new_bias = torch.zeros(new_out, device=bias.device)
            layer.bias = nn.Parameter(torch.cat([bias, new_bias], dim=0))
        self.history.append(ExpansionEvent(layer_name=str(layer), new_neurons=new_out))
