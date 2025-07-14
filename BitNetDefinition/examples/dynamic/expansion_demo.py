try:
    from torch import nn
except ImportError:
    raise SystemExit("PyTorch not installed")

from src.models.dynamic.expansion import DynamicExpansion

layer = nn.Linear(4, 2)
expander = DynamicExpansion(layer)
print("Before:", layer.weight.shape)
expander.expand_linear(layer, 1)
print("After:", layer.weight.shape)
