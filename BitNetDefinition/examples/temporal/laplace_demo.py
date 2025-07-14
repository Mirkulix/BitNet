try:
    import torch
except ImportError:
    raise SystemExit("PyTorch not installed")

from src.models.temporal.laplace import LaplaceLayer

layer = LaplaceLayer(3, 2)
x = torch.randn(1, 3)
print(layer(x))
