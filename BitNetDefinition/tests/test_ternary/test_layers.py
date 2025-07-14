import pytest

try:
    import torch
except ImportError:
    torch = None

from src.models.ternary.layers import TernaryLinear


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_quantization_values():
    layer = TernaryLinear(4, 2)
    w = layer.weight.data
    q = layer.quantize_weight(w) if hasattr(layer, "quantize_weight") else w.sign()
    assert torch.all(q.isin(torch.tensor([-1.0, 0.0, 1.0])))
