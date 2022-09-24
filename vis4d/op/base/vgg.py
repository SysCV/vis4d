"""Residual networks for classification."""

from typing import List

import torch
import torchvision.models.vgg as _vgg
from torchvision.models._utils import IntermediateLayerGetter

from .base import BaseModel


class VGG(BaseModel):
    """Wrapper for torch vision resnet backbones."""

    def __init__(
        self,
        vgg_name: str,
        pretrained: bool = False,
    ):
        """Initialize the VGG base model from torchvision.

        Args:
            vgg_name (str): name of the VGG variant. Choices in ["vgg11",
                "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn",
                "vgg19_bn"].
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

        vgg = _vgg.__dict__[vgg_name](pretrained=pretrained)
        use_bn = vgg_name[-3:] == "_bn"
        self._out_channels = []
        returned_layers = []
        last_channel = None
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

        return_layers = {str(v): str(i) for i, v in enumerate(returned_layers)}
        self.body = IntermediateLayerGetter(
            vgg.features, return_layers=return_layers
        )
        self.name = vgg_name

    @property
    def out_channels(self) -> List[int]:
        """Get the number of channels for each level of feature pyramid.

        Returns:
            List[int]: number of channels
        """
        return self._out_channels

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """VGG feature forward without classification head.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (List[torch.Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling raio of 2^index.
            fp[0] and fp[1] is a reference to the input images. The last
            feature map downsamples the input image by 64.
        """
        return [images, images, *self.body(images).values()]
