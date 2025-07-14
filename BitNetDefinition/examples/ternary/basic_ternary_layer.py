from src.models.ternary.layers import TernaryLinear

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch not installed")

layer = TernaryLinear(4, 2)
x = torch.randn(1, 4)
print(layer(x))
