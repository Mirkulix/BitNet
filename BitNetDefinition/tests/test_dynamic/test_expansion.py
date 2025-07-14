import pytest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from src.models.dynamic.expansion import DynamicExpansion


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_expand_linear_increases_neurons():
    layer = nn.Linear(4, 2)
    expander = DynamicExpansion(layer)
    expander.expand_linear(layer, 1)
    assert layer.weight.shape[0] == 3
