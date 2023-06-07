"""Transformer layer.

Modified from timm (https://github.com/huggingface/pytorch-image-models).
"""
from __future__ import annotations

import torch
from torch import nn

from .attention import Attention
from .drop import DropPath
from .mlp import TransformerBlockMLP


class _LayerScale(nn.Module):
    """Layer scaler."""

    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ):
        """Init layer scaler.

        Args:
            dim (int): Input tensor's dimension.
            init_values (float, optional): Initial values for layer scale.
                Defaults to 1e-5.
            inplace (bool, optional): If to do the operation in-place. Defaults
                to False.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        """Forward pass."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerBlock(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: tuple[float, float] | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """Init transformer block.

        Args:
            dim (int): Input tensor's dimension.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding
                dim. Defaults to 4.0.
            qkv_bias (bool, optional): If to add bias to qkv. Defaults to
                False.
            drop (float, optional): Dropout rate for attention and projection.
                Defaults to 0.0.
            attn_drop (float, optional): Dropout rate for attention. Defaults
                to 0.0.
            init_values (tuple[float, float] | None, optional): Initial values
                for layer scale. Defaults to None.
            drop_path (float, optional): Dropout rate for drop path. Defaults
                to 0.0.
            act_layer (nn.Module, optional): Activation layer. Defaults to
                nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to
                nn.LayerNorm.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            _LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = TransformerBlockMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            _LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            data (torch.Tensor): Input tensor of shape (B, N, dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim).
        """
        return self._call_impl(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
