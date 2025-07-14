"""Simple Laplace transform based layer."""

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


class LaplaceLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        if torch is None:
            raise ImportError("PyTorch required for LaplaceLayer")
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.matmul(x, self.weight.T)
