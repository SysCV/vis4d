"""Drop Path. Reference from
`https://github.com/huggingface/pytorch-image-models/blob/timm/layers/drop.py`_.
"""

import torch
from torch import nn


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
):
    """Drop paths (stochastic depth) per sample of residual blocks.

    Args:
        x: (torch.Tensor) Input tensor.
        drop_prob: (float) Drop probability. Defaults to 0.0.
        training: (bool) Whether in training mode or not. Defaults to False.
        scale_by_keep: (bool) Whether to scale by keep probability. Defaults to
            True.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample of residual blocks.."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initialize DropPath.

        Args:
            drop_prob (float): Drop probability. Defaults to 0.0.
            scale_by_keep (bool): Whether to scale by keep probability.
                Defaults to True.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
