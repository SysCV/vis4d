"""Spatial Cross Attention Module for BEVFormer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from vis4d.op.layer.ms_deform_attn import (
    MSDeformAttentionFunction,
    is_power_of_2,
    ms_deformable_attention_cpu,
)
from vis4d.op.layer.weight_init import constant_init, xavier_init


class SpatialCrossAttention(nn.Module):
    """An attention module used in BEVFormer."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        dropout: float = 0.1,
        deformable_attention: MSDeformableAttention3D | None = None,
    ) -> None:
        """Init.

        Args:
            embed_dims (int): The embedding dimension of Attention. Default:
                256.
            num_cams (int): The number of cameras. Default: 6.
            dropout (float): A Dropout layer on `inp_residual`. Default: 0.1.
            deformable_attention (MSDeformableAttention3D, optional):
                The deformable attention module. Default: None. If None,
                we will use `MSDeformableAttention3D` with default
                parameters.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = (
            deformable_attention or MSDeformableAttention3D()
        )
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self) -> None:
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        bev_mask: Tensor,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """Forward Function of Detr3DCrossAtten.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            reference_points (Tensor):  The normalized reference points with
                shape (bs, num_query, 4), all elements is range in [0, 1],
                top-left (0,0), bottom-right (1, 1), including padding area.
                Or (N, Length_{query}, num_levels, 4), add additional two
                dimensions is (w, h) to form reference boxes.
            value (Tensor): The value tensor with shape `(num_key, bs,
                embed_dims)`. (B, N, C, H, W)
            spatial_shapes (Tensor): Spatial shape of features in different
                level. With shape (num_levels, 2), last dimension represent
                (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            bev_mask (Tensor): The mask of BEV features with shape
                (num_query, bs, num_levels, h, w).
            query_pos (Tensor): The positional encoding for `query`. Default
                None.

        Returns:
            Tensor: Forwarded results with shape [num_query, bs, embed_dims].
        """
        inp_residual = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        bs = query.shape[0]
        d = reference_points.shape[3]

        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max(len(each) for each in indexes)

        # Each camera only interacts with its corresponding BEV queries.
        # This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims]
        )
        reference_points_rebatch = reference_points.new_zeros(
            [bs, self.num_cams, max_len, d, 2]
        )

        for j in range(bs):
            for i, _reference_points in enumerate(reference_points):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query[
                    j, index_query_per_img
                ]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = (
                    _reference_points[j, index_query_per_img]
                )

        _, l, bs, _ = value.shape

        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims
        )

        queries = self.deformable_attention(
            query=queries_rebatch.view(
                bs * self.num_cams, max_len, self.embed_dims
            ),
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, d, 2
            ),
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[
                    j, i, : len(index_query_per_img)
                ]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


class MSDeformableAttention3D(nn.Module):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        im2col_step: int = 64,
        batch_first: bool = True,
    ) -> None:
        """Init.

        Args:
            embed_dims (int): The embedding dimension of Attention. Default:
                256.
            num_heads (int): Parallel attention heads. Default: 64.
            num_levels (int): The number of feature map used in
                Attention. Default: 4.
            num_points (int): The number of sampling points for each query in
                each head. Default: 4.
            im2col_step (int): The step used in image_to_column.
                Default: 64.
            batch_first (bool): Key, Query and Value are shape of (batch, n,
                embed_dim) or (n, batch, embed_dim). Default to True.
        """
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        self.batch_first = batch_first

        is_power_of_2(embed_dims // num_heads)

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.mul(
            torch.arange(self.num_heads, dtype=torch.float32),
            (2.0 * math.pi / self.num_heads),
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)

    def forward(  # pylint: disable=duplicate-code
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            query (Tensor): Query of Transformer with shape (bs, num_query,
                embed_dims).
            reference_points (Tensor):  The normalized reference points with
                shape (bs, num_query, num_levels, 2), all elements is range in
                [0, 1], top-left (0,0), bottom-right (1, 1), including padding
                area. Or (N, Length_{query}, num_levels, 4), add additional two
                dimensions is (w, h) to form reference boxes.
            value (Tensor): The value tensor with shape `(bs, num_key,
                embed_dims)`.
            spatial_shapes (Tensor): Spatial shape of features in different
                levels. With shape (num_levels, 2), last dimension represents
                (h, w).
            level_start_index (Tensor): The start index of each level. A tensor
                has shape ``(num_levels, )`` and can be represented as [0,
                h_0*w_0, h_0*w_0+h_1*w_1, ...].
            key_padding_mask (Tensor): ByteTensor for value, with shape [bs,
                num_key].
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        attention_weights = attention_weights.softmax(-1)

        # bs, num_query, num_heads, num_levels, num_all_points
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # For each BEV query, it owns `num_z_anchors` in 3D space that
        # having different heights. After proejcting, each BEV query has
        # `num_z_anchors` reference points in each 2D image. For each
        # referent point, we sample `num_points` sampling points.
        # For `num_z_anchors` reference points, it has overall `num_points
        # * num_z_anchors` sampling points.
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )

            bs, num_query, num_z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = (
                sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points,
                xy,
            ) = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points // num_z_anchors,
                num_z_anchors,
                xy,
            )
            sampling_locations = reference_points + sampling_offsets
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_points,
                num_z_anchors,
                xy,
            ) = sampling_locations.shape
            assert num_all_points == num_points * num_z_anchors

            # bs, num_query, num_heads, num_levels, num_all_points, 2
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 , but get "
                + f"{reference_points.shape[-1]} instead."
            )

        if torch.cuda.is_available() and value.is_cuda:
            output = MSDeformAttentionFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = ms_deformable_attention_cpu(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
