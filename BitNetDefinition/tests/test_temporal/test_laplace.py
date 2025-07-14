import pytest

try:
    import torch
except ImportError:
    torch = None

from src.models.temporal.laplace import LaplaceLayer


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_laplace_shape():
    layer = LaplaceLayer(3, 2)
    x = torch.randn(5, 3)
    out = layer(x)
    assert out.shape == (5, 2)
