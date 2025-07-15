"""Residual networks base model.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

from collections.abc import Sequence

import torchvision.models.resnet as _resnet
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.checkpoint import checkpoint

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.typing import ArgsType
from vis4d.op.layer.util import build_conv_layer, build_norm_layer
from vis4d.op.layer.weight_init import constant_init, kaiming_init

from .base import BaseModel


class BasicBlock(nn.Module):
    """BasicBlock."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        use_checkpoint: bool = False,
        with_dcn: bool = False,
        norm: str = "BatchNorm2d",
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        assert style in {"pytorch", "caffe"}  # No effect for BasicBlock
        assert not with_dcn, "DCN is not supported for BasicBlock."
        self.conv1 = build_conv_layer(
            inplanes,
            planes,
            3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn1 = build_norm_layer(norm, planes)
        self.conv2 = build_conv_layer(planes, planes, 3, padding=1, bias=False)
        self.bn2 = build_norm_layer(norm, planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.use_checkpoint = use_checkpoint

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""

        def _inner_forward(x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.use_checkpoint and x.requires_grad:
            out = checkpoint(_inner_forward, x, use_reentrant=True)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        use_checkpoint: bool = False,
        with_dcn: bool = False,
        norm: str = "BatchNorm2d",
    ) -> None:
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.use_checkpoint = use_checkpoint

        assert style in {"pytorch", "caffe"}
        if style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.conv1 = build_conv_layer(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )
        self.bn1 = build_norm_layer(norm, planes)

        self.conv2 = build_conv_layer(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            use_dcn=with_dcn,
        )
        self.bn2 = build_norm_layer(norm, planes)

        self.conv3 = build_conv_layer(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = build_norm_layer(norm, planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""

        def _inner_forward(x: Tensor) -> Tensor:
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.use_checkpoint and x.requires_grad:
            out = checkpoint(_inner_forward, x, use_reentrant=True)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(BaseModel):
    """ResNet BaseModel."""

    arch_settings = {
        "resnet18": (18, BasicBlock, (2, 2, 2, 2)),
        "resnet34": (34, BasicBlock, (3, 4, 6, 3)),
        "resnet50": (50, Bottleneck, (3, 4, 6, 3)),
        "resnet101": (101, Bottleneck, (3, 4, 23, 3)),
        "resnet152": (152, Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        resnet_name: str,
        in_channels: int = 3,
        stem_channels: int | None = None,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        style: str = "pytorch",
        deep_stem: bool = False,
        avg_down: bool = False,
        trainable_layers: int = 5,
        norm: str = "BatchNorm2d",
        norm_frozen: bool = True,
        stages_with_dcn: Sequence[bool] = (False, False, False, False),
        replace_stride_with_dilation: Sequence[bool] = (False, False, False),
        use_checkpoint: bool = False,
        zero_init_residual: bool = True,
        pretrained: bool = False,
        weights: None | str = None,
    ) -> None:
        """Create ResNet.

        Args:
            resnet_name (str): Name of the ResNet variant.
            in_channels (int): Number of input image channels. Default: 3.
            stem_channels (int | None): Number of stem channels. If not
                specified, it will be the same as `base_channels`. Default:
                None.
            base_channels (int): Number of base channels of res layer. Default:
                64.
            num_stages (int): Resnet stages. Default: 4.
            strides (Sequence[int]): Strides of the first block of each stage.
                Default: (1, 2, 2, 2).
            dilations (Sequence[int]): Dilation of each stage. Default: (1, 1,
                1, 1)
            style (str): `pytorch` or `caffe`. If set to "pytorch", the
                stride-two layer is the 3x3 conv layer, otherwise the
                stride-two layer is the first 1x1 conv layer. Default: pytorch.
            deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
                Default: False.
            avg_down (bool): Use AvgPool instead of stride conv when
                downsampling in the bottleneck. Default: False.
            trainable_layers (int, optional): Number layers for training or
                fine-tuning. 5 means all the layers can be fine-tuned. Defaults
                to 5.
            norm (str): Normalization layer str. Default: BatchNorm2d, which
                means using `nn.BatchNorm2d`.
            norm_frozen (bool): Whether to set norm layers to eval mode. It
                freezes running stats (mean and var). Note: Effect on
                Batch Norm and its variants only.
            stages_with_dcn (Sequence[bool]): Indices of stages with deformable
                convolutions. Default: (False, False, False, False).
            replace_stride_with_dilation (Sequence[bool]): Whether to replace
                stride with dilation. Default: (False, False, False).
            use_checkpoint (bool): Use checkpoint or not. Using checkpoint will
                save some memory while slowing down the training speed.
                Default: False.
            zero_init_residual (bool): Whether to use zero init for last norm
                layer in resblocks to let them behave as identity.
                Default: True.
            pretrained (bool): Whether to load pretrained weights. Default:
                False.
            weights (str, optional): model pretrained path. Default: None
        """
        super().__init__()
        self._norm = norm

        self.zero_init_residual = zero_init_residual
        if resnet_name not in self.arch_settings:
            raise KeyError(f"invalid architecture {resnet_name} for ResNet")
        self.name = resnet_name
        self.deep_stem = deep_stem
        self.trainable_layers = trainable_layers

        self.use_checkpoint = use_checkpoint
        self.norm_frozen = norm_frozen

        depth, self.block, stage_blocks = self.arch_settings[resnet_name]
        assert isinstance(depth, int)

        self.depth = depth
        stem_channels = stem_channels or base_channels

        assert 4 >= num_stages >= 1
        assert len(strides) == len(dilations) == num_stages

        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            if i > 0 and replace_stride_with_dilation[i - 1]:
                dilation = strides[i]
                stride = 1
            else:
                stride = strides[i]
                dilation = dilations[i]
            planes = base_channels * 2**i
            res_layer = self._make_res_layer(
                block=self.block,  # type: ignore
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=style,
                avg_down=avg_down,
                use_checkpoint=use_checkpoint,
                with_dcn=stages_with_dcn[i],
            )
            self.inplanes = planes * self.block.expansion  # type: ignore
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        if pretrained:
            if weights is None:
                # default loading the imagenet-1k v1 pre-trained model weights
                weights = _resnet.__dict__[
                    f"ResNet{depth}_Weights"
                ].IMAGENET1K_V1.url

            load_model_checkpoint(self, weights)
        else:
            self._init_weights()

        self._freeze_stages()

    def _init_weights(self) -> None:
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and isinstance(
                    m.bn3.weight, nn.Parameter
                ):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and isinstance(
                    m.bn2.weight, nn.Parameter
                ):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self._norm, stem_channels // 2),
                nn.ReLU(inplace=True),
                build_conv_layer(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self._norm, stem_channels // 2),
                nn.ReLU(inplace=True),
                build_conv_layer(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self._norm, stem_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = build_conv_layer(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.bn1 = build_norm_layer(self._norm, stem_channels)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_res_layer(
        self,
        block: BasicBlock | Bottleneck,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        style: str,
        avg_down: bool,
        use_checkpoint: bool,
        with_dcn: bool,
    ) -> nn.Sequential:
        """Pack all blocks in a stage into a ``ResLayer``."""
        layers: list[BasicBlock | Bottleneck] = []
        downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample_list: list[nn.AvgPool2d | nn.Module] = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample_list.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    )
                )
            downsample_list.extend(
                [
                    build_conv_layer(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    build_norm_layer(self._norm, planes * block.expansion),
                ]
            )
            downsample = nn.Sequential(*downsample_list)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                use_checkpoint=use_checkpoint,
                with_dcn=with_dcn,
                norm=self._norm,
            )
        )
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    style=style,
                    use_checkpoint=use_checkpoint,
                    with_dcn=with_dcn,
                    norm=self._norm,
                )
            )
        return nn.Sequential(*layers)

    def _freeze_stages(self) -> None:
        """Freeze stages param and norm stats."""
        if self.trainable_layers < 5:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.bn1.eval()
                for m in (self.conv1, self.bn1):
                    for param in m.parameters():
                        param.requires_grad = False

            for i in range(1, 5 - self.trainable_layers):
                m = getattr(self, f"layer{i}")
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> ResNet:
        """Override the train mode for the model."""
        super().train(mode)
        self._freeze_stages()

        if mode and self.norm_frozen:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self

    @property
    def out_channels(self) -> list[int]:
        """Get the number of channels for each level of feature pyramid.

        Returns:
            list[int]: number of channels
        """
        if self.name in {"resnet18", "resnet34"}:
            # channels = [3, 3] + [64 * 2**i for i in range(4)]
            channels = [3, 3, 64, 128, 256, 512]
        else:
            # channels = [3, 3] + [256 * 2**i for i in range(4)]
            channels = [3, 3, 256, 512, 1024, 2048]
        return channels

    def forward(self, images: Tensor) -> list[Tensor]:
        """Forward function.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (list[torch.Tensor]): The output feature pyramid. The list index
                represents the level, which has a downsampling raio of 2^index.
                fp[0] and fp[1] is a reference to the input images and
                torchvision resnet downsamples the feature maps by 4 directly.
                The last feature map downsamples the input image by 64 with a
                pooling layer on the second last map.
        """
        if self.deep_stem:
            x = self.stem(images)
        else:
            x = self.conv1(images)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = [images, images]
        for _, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)
        return outs


class ResNetV1c(ResNet):
    """ResNetV1c variant with a deeper stem.

    Compared with default ResNet, ResNetV1c replaces the 7x7 conv in the input
    stem with three 3x3 convs. For more details please refer to `Bag of Tricks
    for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`.
    """

    model_urls = {
        "resnet50_v1c": (
            "https://download.openmmlab.com/pretrain/third_party/"
            "resnet50_v1c-2cccc1ad.pth"
        ),
        "resnet101_v1c": (
            "https://download.openmmlab.com/pretrain/third_party/"
            "resnet101_v1c-e67eebb6.pth"
        ),
    }

    def __init__(
        self,
        resnet_name: str,
        pretrained: bool = False,
        weights: str | None = None,
        **kwargs: ArgsType,
    ):
        """Initialize ResNetV1c.

        Args:
            resnet_name (str): Name of the resnet model.
            pretrained (bool, optional): Whether to load ImageNet pre-trained
                weights. Defaults to False.
            weights (str, optional): Path to custom pretrained weights.
            **kwargs: Arguments for ResNet.
        """
        assert resnet_name in {
            "resnet18_v1c",
            "resnet34_v1c",
            "resnet50_v1c",
            "resnet101_v1c",
        }
        if pretrained and weights is None:
            assert resnet_name in {
                "resnet50_v1c",
                "resnet101_v1c",
            }, "Only resnet50_v1c and resnet101_v1c have pretrained weights."
            weights = self.model_urls[resnet_name]

        super().__init__(
            resnet_name[:-4],
            deep_stem=True,
            pretrained=pretrained,
            weights=weights,
            **kwargs,
        )
