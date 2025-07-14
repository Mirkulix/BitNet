"""Training helpers for ternary networks."""

try:
    import torch
    from torch.nn.utils import clip_grad_norm_
except ImportError:  # pragma: no cover - torch not installed
    torch = None
    clip_grad_norm_ = None


def clip_gradients(parameters, max_norm: float = 1.0):
    if clip_grad_norm_ is None:
        raise ImportError("PyTorch required for clip_gradients")
    clip_grad_norm_(parameters, max_norm)
