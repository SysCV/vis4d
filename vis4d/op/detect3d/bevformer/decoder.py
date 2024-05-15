"""BEVFormer decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from vis4d.op.layer.attention import MultiheadAttention
from vis4d.op.layer.ms_deform_attn import (
    MSDeformAttentionFunction,
    is_power_of_2,
    ms_deformable_attention_cpu,
)
from vis4d.op.layer.transformer import FFN, inverse_sigmoid
from vis4d.op.layer.weight_init import constant_init, xavier_init


class BEVFormerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer."""

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        return_intermediate: bool = True,
    ) -> None:
        """Init.

        Args:
            num_layers (int): The number of decoder layers. Default: 6.
            embed_dims (int): The embedding dimension. Default: 256.
            return_intermediate (bool): Whether to return intermediate
                results. Default: True.
        """
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = nn.ModuleList(
            [
                (BEVFormerDecoderLayer(embed_dims=embed_dims))
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        query_pos: Tensor,
        reg_branches: list[nn.Module],
    ) -> tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims).
            value (Tensor): Input value with shape (bs, num_query, embed_dims).
            reference_points (Tensor): The reference points of offset. In shape
                (bs, num_query, 4) when as_two_stage, otherwise has shape (bs,
                num_query, 2).
            spatial_shapes (Tensor): The spatial shapes of feature maps.
            level_start_index (Tensor): The start index of each level.
            query_pos (Tensor): The query position embedding.
            reg_branches: (list[nn.Module]): Used for refining the regression
                results.

        Returns:
            tuple[Tensor, Tensor]: The output of the decoder with reference
                points. If return_intermediate is True, the output and
                reference points of each layer will be stacked and return.
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # BS, NUM_QUERY, NUM_LEVEL, 2
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                query_pos=query_pos,
            )
            output = output.permute(1, 0, 2)

            tmp = reg_branches[lid](output)

            assert reference_points.shape[-1] == 3
            new_reference_points = torch.zeros_like(reference_points)
            new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                reference_points[..., :2]
            )
            new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(
                reference_points[..., 2:3]
            )

            new_reference_points = new_reference_points.sigmoid()

            reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points
            )

        return output, reference_points


class BEVFormerDecoderLayer(nn.Module):
    """Implements decoder layer in DETR transformer."""

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 512,
        drop_out: float = 0.1,
    ) -> None:
        """Init.

        Args:
            embed_dims (int): The embedding dimension.
            feedforward_channels (int): The hidden dimension of FFNs.
            drop_out (float): The dropout rate of FFNs.
        """
        super().__init__()
        self.attentions = nn.ModuleList()

        self.attentions.append(
            MultiheadAttention(
                embed_dims=embed_dims,
                num_heads=8,
                attn_drop=0.1,
                proj_drop=0.1,
            )
        )
        self.attentions.append(
            DecoderCrossAttention(embed_dims=embed_dims, num_levels=1)
        )

        self.embed_dims = embed_dims

        self.ffns = nn.ModuleList()
        self.ffns.append(
            FFN(
                embed_dims=self.embed_dims,
                feedforward_channels=feedforward_channels,
                dropout=drop_out,
            )
        )

        self.norms = nn.ModuleList()
        for _ in range(3):
            self.norms.append(nn.LayerNorm(self.embed_dims))

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            reference_points (Tensor): The reference points of offset. In shape
                (bs, num_query, 4) when as_two_stage, otherwise has shape (bs,
                num_query, 2).
            value (Tensor, optional): The input value, has shape (bs, num_keys,
                dim).
            spatial_shapes (Tensor): The spatial shapes of feature maps.
            level_start_index (Tensor): The start index of each level.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.attentions[0](
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
        )

        query = self.norms[0](query)

        query = self.attentions[1](
            query=query,
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            query_pos=query_pos,
        )

        query = self.norms[1](query)

        query = self.ffns[0](query)

        query = self.norms[2](query)

        return query


class DecoderCrossAttention(nn.Module):
    """Custom Multi-Scale Deformable Attention."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
    ) -> None:
        """Initialization.

        Args:
            embed_dims (int): The embedding dimension of Attention.
                Default: 256.
            num_heads (int): Parallel attention heads. Default: 8.
            num_levels (int): The number of feature map used in Attention.
                Default: 4.
            num_points (int): The number of sampling points for each query in
                each head. Default: 4.
            im2col_step (int): The step used in image_to_column.
                Default: 64.
            dropout (float): A Dropout layer on `inp_identity`.
                Default: 0.1.
            batch_first (bool): Key, Query and Value are shape of (batch, n,
                embed_dim) or (n, batch, embed_dim). Default to False.
        """
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        self.dropout = nn.Dropout(dropout)
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
        self.output_proj = nn.Linear(embed_dims, embed_dims)
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
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(  # pylint: disable=duplicate-code
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
        identity: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            query (Tensor): Query of Transformer with shape (num_query, bs,
                embed_dims).
            reference_points (Tensor):  The normalized reference points with
                shape (bs, num_query, num_levels, 2), all elements is range in
                [0, 1], top-left (0,0), bottom-right (1, 1), including padding
                area. or (N, Length_{query}, num_levels, 4), add additional two
                dimensions is (w, h) to form reference boxes.
            value (Tensor): The value tensor with shape (num_key, bs,
                embed_dims).
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query ,embed_dims)
        if not self.batch_first:
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

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
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

        output = self.output_proj(output)

        # (num_query, bs ,embed_dims)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
