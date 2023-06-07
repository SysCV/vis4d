"""Attention layer.

Modified from timm (https://github.com/huggingface/pytorch-image-models).
"""
from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Init attention layer.

        Args:
            dim (int): Input tensor's dimension.
            num_heads (int, optional): Number of attention heads. Defaults to
                8.
            qkv_bias (bool, optional): If to add bias to qkv. Defaults to
                False.
            attn_drop (float, optional): Dropout rate for attention. Defaults
                to 0.0.
            proj_drop (float, optional): Dropout rate for projection. Defaults
                to 0.0.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies the layer.

        Args:
            data (torch.Tensor): Input tensor of shape (B, N, dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self._call_impl(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, num_samples, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                num_samples,
                3,
                self.num_heads,
                dim // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(
            0
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_samples, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
