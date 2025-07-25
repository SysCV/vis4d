"""Positional encoding for transformer.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .weight_init import uniform_init


class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(
        self,
        num_feats: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ) -> None:
        """Initialization for `SinePositionalEncoding`.

        Args:
            num_feats (int): The feature dimension for each position
                along x-axis or y-axis. Note the final returned dimension
                for each position is 2 times of this value.
            temperature (int, optional): The temperature used for scaling
                the position embedding. Defaults to 10000.
            normalize (bool, optional): Whether to normalize the position
                embedding. Defaults to False.
            scale (float, optional): A scale factor that scales the position
                embedding. The scale will be used only when normalize is True.
                Defaults to 2*pi.
            eps (float, optional): A value added to the denominator for
                numerical stability. Defaults to 1e-6.
            offset (float, optional): offset add to embed when do the
                normalization. Defaults to 0.
        """
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(
        self, mask: Tensor | None, inputs: Tensor | None = None
    ) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor | None): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w]. If None, it means single
                image or batch image with no padding.
            inputs (Tensor | None): The input tensor. It mask is None, this
                input tensor is required to get the shape of the input image.

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        if mask is not None:
            # For convenience of exporting to ONNX, it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            b, h, w = mask.size()
            device = mask.device
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            assert isinstance(inputs, Tensor)
            b, _, h, w = inputs.shape
            device = inputs.device
            x_embed = torch.arange(
                1, w + 1, dtype=torch.float32, device=device
            )
            x_embed = x_embed.view(1, 1, -1).repeat(b, h, 1)
            y_embed = torch.arange(
                1, h + 1, dtype=torch.float32, device=device
            )
            y_embed = y_embed.view(1, -1, 1).repeat(b, 1, w)
        if self.normalize:
            y_embed = (
                (y_embed + self.offset)
                / (y_embed[:, -1:, :] + self.eps)
                * self.scale
            )
            x_embed = (
                (x_embed + self.offset)
                / (x_embed[:, :, -1:] + self.eps)
                * self.scale
            )
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).view(b, h, w, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).view(b, h, w, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights."""

    def __init__(
        self, num_feats: int, row_num_embed: int = 50, col_num_embed: int = 50
    ) -> None:
        """Initialization for LearnedPositionalEncoding.

        Args:
            num_feats (int): The feature dimension for each position
                along x-axis or y-axis. The final returned dimension for
                each position is 2 times of this value.
            row_num_embed (int, optional): The dictionary size of row
                embeddings. Defaults to 50.
            col_num_embed (int, optional): The dictionary size of col
                embeddings. Defaults to 50.
        """
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of position embedding."""
        uniform_init(self.row_embed, lower=0, upper=1)
        uniform_init(self.col_embed, lower=0, upper=1)

    def forward(self, mask: Tensor) -> Tensor:
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat(
                (
                    x_embed.unsqueeze(0).repeat(h, 1, 1),
                    y_embed.unsqueeze(1).repeat(1, w, 1),
                ),
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos
