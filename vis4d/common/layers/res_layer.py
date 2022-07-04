from typing import Optional, Union, Tuple
import torch
from torch import nn
from .conv2d import Conv2d


class ResidualBlock(nn.Module):  # type: ignore
    """ Basic Residual block.

        x -> [Conv1] -> [Activation] -> [Norm] -> [Conv2] ->  [Norm] -> (+) -> [Activation] ->
          |                                                             |
          --------------------------------------------------------------

        Note that if stride or padding is set in a way that downsamples the resulting features,
        the input signal is convolved with an additional convolutional layer in order to
        downsample it to the right dimension before performing the addition.
    """

    def __init__(
        self,
        conv_in_dim: int,
        conv_out_dim: int,
        conv_has_bias: bool = False,
        stride: int = 1,
        kernel_size: Union[Tuple[int,int],int] = 3,
        padding: Union[Tuple[int,int], int,str] = 1,
        activation_cfg: str = "ReLU",
        norm_cfg: Optional[str] = "BatchNorm2d",
    ):
        """
        Args:
            conv_in_dim: int, number of input channels
            conv_out_dim: int, number of output channels. If != conv_in_dim the features in the identity branch
                         will be downsampled using an additional convolution
            conv_has_bias: bool, if true adds learnable bias to convolutional layers
            stride: int, stride value for the convolution operations.
            kernel_size: Kernel Size. See conv2D
            padding: Padding. See torch.nn.Conv2D
            activation_cfg: str, the name of the activation function (must be part of the torch.nn namespace)
            norm_cfg: str, the norm that should be used (must be part of the torch.nn namespace).
        """
        super().__init__()

        self.apply_downsampling = conv_in_dim != conv_out_dim
        self.stride = stride
        if norm_cfg is not None:
            norm = getattr(nn, norm_cfg)
        else:
            norm = None  # pragma: no cover
        activation_func = getattr(nn, activation_cfg)

        self.conv1 = Conv2d(
            conv_in_dim,
            conv_out_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=conv_has_bias,
            norm=norm(conv_out_dim) if norm is not None else norm,
            activation=activation_func(inplace=True)
        )
        self.conv2 = Conv2d(
            conv_out_dim,
            conv_out_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=conv_has_bias,
            norm=norm(conv_out_dim) if norm is not None else norm,
        )

        if self.apply_downsampling:
            self.downsample = Conv2d(
                conv_in_dim,
                conv_out_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=conv_has_bias,
                padding=padding,
                norm=norm(conv_out_dim) if norm is not None else norm,
            )
        self.activation = activation_func(inplace=True)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        identity = input_x
        out = self.conv1(input_x)
        out = self.conv2(out)

        # This means that the identity can not just be added as the dimensionality changes
        if self.apply_downsampling:
            identity = self.downsample(input_x)
        out += identity

        out = self.activation(out)
        return out
