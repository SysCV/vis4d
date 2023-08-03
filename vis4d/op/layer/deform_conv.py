"""Wrapper for deformable convolution."""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d
from vis4d_cuda_ops import (
    modulated_deform_conv_backward,
    modulated_deform_conv_forward,
)

from vis4d.common.logging import rank_zero_info


class DeformConv(nn.Module):
    """Deformable Convolution operator.

    Includes batch normalization and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        bn_momentum: float = 0.1,
    ) -> None:
        """Creates an instance of the class.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (tuple[int, int], optional): Size of convolutional
                kernel. Defaults to (3, 3).
            stride (int, optional): Stride of convolutional layer. Defaults to
                1.
            padding (int, optional): Padding of convolutional layer. Defaults
                to 1.
            dilation (int, optional): Dilation of convolutional layer. Defaults
                to 1.
            groups (int, optional): Number of deformable groups. Defaults to 1.
            bias (bool, optional): Whether to use bias in convolutional layer.
                Defaults to True.
            bn_momentum (float, optional): Momentum of batch normalization.
                Defaults to 0.1.
        """
        super().__init__()
        self.conv_offset = nn.Conv2d(
            in_channels,
            groups * 3 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights of offset conv layer."""
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            if self.conv_offset.bias is not None:
                self.conv_offset.bias.data.zero_()

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv_offset(input_x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        input_x = self.deform_conv(input_x, offset, mask)
        input_x = self.actf(input_x)
        return input_x


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class _ModulatedDeformConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
    ):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError(
                "Deformable Conv is not supported on CPUs!"
            )
        if (
            weight.requires_grad
            or mask.requires_grad
            or offset.requires_grad
            or input.requires_grad
        ):
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            _ModulatedDeformConv._infer_shape(ctx, input, weight)
        )
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError(
                "Deformable Conv is not supported on CPUs!"
            )
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias,
        )
        if not ctx.with_bias:
            grad_bias = None

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (
            height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)
        ) // ctx.stride + 1
        width_out = (
            width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)
        ) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = _ModulatedDeformConv.apply


class ModulatedDeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        """
        Modulated deformable convolution from :paper:`deformconv2`.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(
                out_channels, in_channels // groups, *self.kernel_size
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, offset, mask):
        if x.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:],
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                    self.stride,
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr


class ModulatedDeformConv2dPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups
            * 3
            * self.kernel_size[0]
            * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()

    def init_weights(self) -> None:
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return super().forward(x, offset, mask)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
            if (
                prefix + "conv_offset.weight" not in state_dict
                and prefix[:-1] + "_offset.weight" in state_dict
            ):
                state_dict[prefix + "conv_offset.weight"] = state_dict.pop(
                    prefix[:-1] + "_offset.weight"
                )
            if (
                prefix + "conv_offset.bias" not in state_dict
                and prefix[:-1] + "_offset.bias" in state_dict
            ):
                state_dict[prefix + "conv_offset.bias"] = state_dict.pop(
                    prefix[:-1] + "_offset.bias"
                )

        if version is not None and version > 1:
            rank_zero_info(
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                "version 2.",
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
