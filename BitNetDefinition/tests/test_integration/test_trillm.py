import pytest

try:
    import torch
except ImportError:
    torch = None

from src.models.trillm import TriLLM


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_trillm_forward():
    model = TriLLM(3, 3, 2)
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 2)
