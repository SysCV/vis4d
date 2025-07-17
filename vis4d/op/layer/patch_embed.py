"""Image to Patch Embedding using Conv2d.

Modified from vision_transformer
(https://github.com/google-research/vision_transformer).
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        """Init PatchEmbed.

        Args:
            img_size (int, optional): Input image's size. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            in_channels (int, optional): Number of input image's channels.
                Defaults to 3.
            embed_dim (int, optional): Patch embedding's dim. Defaults to 768.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to
                None, which means no normalization layer.
            flatten (bool, optional): If to flatten the output tensor.
                Defaults to True.
            bias (bool, optional): If to add bias to the convolution layer.
                Defaults to True.

        Raises:
            ValueError: If the input image's size is not divisible by the patch
                size.
        """
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies the layer.

        Args:
            data (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C), where N is the
                number of patches (N = H * W).
        """
        return self._call_impl(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        _, _, height, width = x.shape
        assert height == self.img_size[0], (
            f"Input image height ({height}) doesn't match model"
            f"({self.img_size})."
        )
        assert width == self.img_size[1], (
            f"Input image width ({width}) doesn't match model"
            f"({self.img_size})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        x = self.norm(x)
        return x
