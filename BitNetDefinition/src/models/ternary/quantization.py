"""Utilities for ternary weight quantization."""

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - torch not installed
    torch = None
    Tensor = None


def ternary_quantize(weight: "Tensor") -> "Tensor":
    if torch is None:
        raise ImportError("PyTorch required for ternary_quantize")
    return weight.sign()
