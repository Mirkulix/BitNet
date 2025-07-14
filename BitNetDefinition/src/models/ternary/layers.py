from __future__ import annotations

try:
    import torch
    from torch import nn, autograd
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch not installed
    torch = None
    nn = object
    F = None


class TernaryQuantizeFn(autograd.Function):
    """Straight-through estimator for ternary quantization."""

    @staticmethod
    def forward(ctx, weight: "torch.Tensor"):
        if torch is None:
            raise ImportError("PyTorch required for TernaryQuantizeFn")
        ctx.save_for_backward(weight)
        return weight.sign()

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        (weight,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[weight.abs() > 1] = 0
        return grad_input


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        if torch is None:
            raise ImportError("PyTorch required for TernaryLinear")
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        quant_w = TernaryQuantizeFn.apply(self.weight)
        return F.linear(x, quant_w, self.bias)
