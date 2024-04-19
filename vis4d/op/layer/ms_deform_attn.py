# pylint: disable=no-name-in-module, abstract-method, arguments-differ
"""Multi-Scale Deformable Attention Module.

Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py) # pylint: disable=line-too-long
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

from vis4d.common.imports import VIS4D_CUDA_OPS_AVAILABLE
from vis4d.common.logging import rank_zero_warn

if VIS4D_CUDA_OPS_AVAILABLE:
    from vis4d_cuda_ops import ms_deform_attn_backward, ms_deform_attn_forward


class MSDeformAttentionFunction(Function):  # pragma: no cover
    """Multi-Scale Deformable Attention Function module."""

    @staticmethod
    def forward(  # type: ignore
        ctx,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ) -> Tensor:
        """Forward pass."""
        if not VIS4D_CUDA_OPS_AVAILABLE:
            raise RuntimeError(
                "MSDeformAttentionFunction requires vis4d cuda ops to run."
            )
        ctx.im2col_step = im2col_step
        output = ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable  # type: ignore
    def backward(
        ctx, grad_output: Tensor
    ) -> tuple[Tensor, None, None, Tensor, Tensor, None]:
        """Backward pass."""
        if not VIS4D_CUDA_OPS_AVAILABLE:
            raise RuntimeError(
                "MSDeformAttentionFunction requires vis4d cuda ops to run."
            )
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        (
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
        ) = ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return (
            grad_value,
            None,
            None,
            grad_sampling_loc,
            grad_attn_weight,
            None,
        )


def ms_deformable_attention_cpu(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape (bs, num_keys, mum_heads,
            embed_dims // num_heads)
        value_spatial_shapes (Tensor): Spatial shape of each feature map, has
            shape (num_levels, 2), last dimension 2 represent (h, w).
        sampling_locations (Tensor): The location of sampling points, has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2), the last
            dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used when
            calculate the attention, has shape (bs ,num_queries, num_heads,
            num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims).
    """
    bs, _, num_heads, embed_dims = value.shape
    (
        _,
        num_queries,
        num_heads,
        num_levels,
        num_points,
        _,
    ) = sampling_locations.shape
    value_list = value.split([h * w for h, w in value_spatial_shapes], dim=1)
    sampling_grids: Tensor = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # bs, h*w, num_heads, embed_dims ->
        # bs, h*w, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, h*w ->
        # bs*num_heads, embed_dims, h, w
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, h, w)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        )
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (
            torch.stack(sampling_value_list, dim=-2).flatten(-2)
            * attention_weights
        )
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def is_power_of_2(number: int) -> None:
    """Check if a number is a power of 2."""
    if (not isinstance(number, int)) or (number < 0):
        raise ValueError(
            f"invalid input for is_power_of_2: {number} (type: {type(number)})"
        )
    if not ((number & (number - 1) == 0) and number != 0):
        rank_zero_warn(
            "You'd better set hidden dimensions in MultiScaleDeformAttention"
            "to make the dimension of each attention head a power of 2, "
            "which is more efficient in our CUDA implementation."
        )


class MSDeformAttention(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        im2col_step: int = 64,
    ) -> None:
        """Creates an instance of the class.

        Args:
            d_model (int): Hidden dimensions.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per attention head per
                feature level.
            im2col_step (int): The step used in image_to_column. Default: 64.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got "
                + f"{d_model} and {n_heads}."
            )

        is_power_of_2(d_model // n_heads)

        self.d_model = d_model
        self.embed_dims = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.im2col_step = im2col_step

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.mul(
            torch.arange(self.n_heads, dtype=torch.float32),
            (2.0 * math.pi / self.n_heads),
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Forward function.

        Args:
            query (Tensor): (n, length_{query}, C).
            reference_points (Tensor): (n, length_{query}, n_levels, 2),
                range in [0, 1], top-left (0,0), bottom-right (1, 1), including
                padding area or (n, length_{query}, n_levels, 4), add
                additional (w, h) to form reference boxes.
            input_flatten (Tensor): (n, \sum_{l=0}^{L-1} H_l \cdot W_l, C).
            input_spatial_shapes (Tensor): (n_levels, 2), [(H_0, W_0),
                (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index (Tensor): (n_levels, ), [0, H_0*W_0,
                H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ...,
                H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask (Tensor): (n, \sum_{l=0}^{L-1} H_l \cdot W_l),
                True for padding elements, False for non-padding elements.

        Retrun
            output (Tensor): (n, length_{query}, C).
        """
        n, len_q, _ = query.shape
        n, len_in, _ = input_flatten.shape
        assert (
            input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]
        ).sum() == len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(
            n, len_in, self.n_heads, self.d_model // self.n_heads
        )
        sampling_offsets = self.sampling_offsets(query).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points
        )
        # n, len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1,
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
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, "
                + f"but get {reference_points.shape[-1]} instead."
            )

        if torch.cuda.is_available() and value.is_cuda:
            output = MSDeformAttentionFunction.apply(
                value,
                input_spatial_shapes,
                input_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = ms_deformable_attention_cpu(
                value,
                input_spatial_shapes,
                sampling_locations,
                attention_weights,
            )

        output = self.output_proj(output)

        return output
