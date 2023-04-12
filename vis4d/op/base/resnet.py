"""Residual networks for classification."""
from __future__ import annotations

import torch
import torchvision.models.resnet as _resnet
from torch.nn.modules.batchnorm import _BatchNorm

from .base import BaseModel


class ResNet(BaseModel):
    """Wrapper for torch vision resnet backbones."""

    def __init__(
        self,
        resnet_name: str,
        trainable_layers: int = 5,
        norm_freezed: bool = True,
        pretrained: bool = False,
        replace_stride_with_dilation: None | list[bool] = None,
    ):
        """Initialize the ResNet base model from torch vision.

        Args:
            resnet_name (str): name of the resnet variant
            trainable_layers (int, optional): Number layers for training or
            fine-tuning. 5 means all the layers can be fine-tuned.
            Defaults to 5.
            norm_freezed (bool, optional): Whether to freeze batch norm.
            Defaults to True.
            pretrained (bool, optional): Whether to load ImageNet
            pre-trained weights. Defaults to False.

        Raises:
            ValueError: trainable_layers should be between 0 and 5
        """
        super().__init__()
        self.name = resnet_name
        self.norm_freezed = norm_freezed

        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = _resnet.__dict__[resnet_name](
            weights=weights,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        # select layers that wont be frozen
        if trainable_layers < 0 or trainable_layers > 5:  # pragma: no cover
            raise ValueError(
                f"Trainable layers should be in the range [0,5], "
                f"got {trainable_layers}"
            )
        self.trainable_layers = trainable_layers

        returned_layers = [1, 2, 3, 4]
        self.return_layers = {
            f"layer{k}": str(v) for v, k in enumerate(returned_layers)
        }

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if self.trainable_layers < 5:
            self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze stages."""
        self.bn1.eval()
        for m in (self.conv1, self.bn1):
            for param in m.parameters():
                param.requires_grad_(False)

        if self.trainable_layers < 4:
            for i in range(1, 5 - self.trainable_layers):
                m = getattr(self, f"layer{i}")
                m.eval()
                for param in m.parameters():
                    param.requires_grad_(False)

    def train(self, mode: bool = True) -> ResNet:
        """Override the train mode for the model."""
        super().train(mode)
        if self.trainable_layers < 5:
            self._freeze_stages()

        if mode and self.norm_freezed:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self

    @property
    def out_channels(self) -> list[int]:
        """Get the number of channels for each level of feature pyramid.

        Returns:
            list[int]: number of channels
        """
        # use static value to be compatible with torch.jit
        if self.name in {"resnet18", "resnet34"}:
            # channels = [3, 3] + [64 * 2**i for i in range(4)]
            channels = [3, 3, 64, 128, 256, 512]
        else:
            # channels = [3, 3] + [256 * 2**i for i in range(4)]
            channels = [3, 3, 256, 512, 1024, 2048]
        return channels

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Torchvision ResNet forward.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (list[torch.Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling raio of 2^index.
            fp[0] and fp[1] is a reference to the input images and torchvision
            resnet downsamples the feature maps by 4 directly. The last feature
            map downsamples the input image by 64 with a pooling layer on the
            second last map.
        """
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = [images, images]
        for _, layer_name in enumerate(self.return_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)
        return outs
