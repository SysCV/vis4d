"""Residual networks for classification."""

from __future__ import annotations

import torch
import torchvision.models.vgg as _vgg
from torchvision.models._utils import IntermediateLayerGetter

from .base import BaseModel


class VGG(BaseModel):
    """Wrapper for torch vision VGG."""

    def __init__(
        self,
        vgg_name: str,
        trainable_layers: None | int = None,
        pretrained: bool = False,
    ):
        """Initialize the VGG base model from torchvision.

        Args:
            vgg_name (str): name of the VGG variant. Choices in ["vgg11",
                "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn",
                "vgg19_bn"].
            trainable_layers (int, optional): Number layers for training or
                fine-tuning. None means all the layers can be fine-tuned.
            pretrained (bool, optional): Whether to load ImageNet
                pre-trained weights. Defaults to False.

        Raises:
            ValueError: The VGG name is not supported
        """
        super().__init__()
        if vgg_name not in [
            "vgg11",
            "vgg13",
            "vgg16",
            "vgg19",
            "vgg11_bn",
            "vgg13_bn",
            "vgg16_bn",
            "vgg19_bn",
        ]:
            raise ValueError("The VGG name is not supported!")

        weights = "IMAGENET1K_V1" if pretrained else None
        vgg = _vgg.__dict__[vgg_name](weights=weights)
        use_bn = vgg_name[-3:] == "_bn"
        self._out_channels: list[int] = []
        returned_layers = []
        last_channel = -1
        layer_counter = 0

        vgg_channels = _vgg.cfgs[
            {"vgg11": "A", "vgg13": "B", "vgg16": "D", "vgg19": "E"}[
                vgg_name[:5]
            ]
        ]
        for channel in vgg_channels:
            if channel == "M":
                returned_layers.append(layer_counter)
                self._out_channels.append(last_channel)
                layer_counter += 1
            else:
                if use_bn:
                    layer_counter += 3
                else:
                    layer_counter += 2
                last_channel = channel

        if trainable_layers is not None:
            for name, parameter in vgg.features.named_parameters():
                layer_ind = int(name.split(".")[0])
                if layer_ind < layer_counter - trainable_layers:
                    parameter.requires_grad_(False)

        return_layers = {str(v): str(i) for i, v in enumerate(returned_layers)}
        self.body = IntermediateLayerGetter(
            vgg.features, return_layers=return_layers
        )
        self.name = vgg_name

    @property
    def out_channels(self) -> list[int]:
        """Get the number of channels for each level of feature pyramid.

        Returns:
            list[int]: number of channels
        """
        return [3, 3, *self._out_channels]

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """VGG feature forward without classification head.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (list[torch.Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling raio of 2^index.
            fp[0] and fp[1] is a reference to the input images. The last
            feature map downsamples the input image by 64.
        """
        return [images, images, *self.body(images).values()]
