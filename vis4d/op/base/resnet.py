"""Residual networks for classification."""

from typing import List

import torch
import torchvision.models.resnet as _resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops

from .base import BaseModel


class ResNet(BaseModel):
    """Wrapper for torch vision resnet backbones."""

    def __init__(
        self,
        resnet_name: str,
        trainable_layers: int = 5,
        freeze_norm: bool = True,
    ):
        """Initiazlie the ResNet base model from torch vision.

        Args:
            resnet_name (str): name of the resnet variant
            trainable_layers (int, optional): Number layers for training or
            fine-tuning. 5 means all the layers can be fine-tuned.
            Defaults to 5.
            freeze_norm (bool, optional): Whether to freeze batch norm.
            Defaults to True.

        Raises:
            ValueError: trainable_layers should be between 0 and 5
        """
        super().__init__()
        resnet = _resnet.__dict__[resnet_name](
            norm_layer=misc_nn_ops.FrozenBatchNorm2d if freeze_norm else None,
        )

        # The code for setting up parametor frozen and layer getter is from
        # torchvision
        # select layers that wont be frozen
        if trainable_layers < 0 or trainable_layers > 5:
            raise ValueError(
                f"Trainable layers should be in the range [0,5], "
                f"got {trainable_layers}"
            )
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
            :trainable_layers
        ]
        if trainable_layers == 5:
            layers_to_train.append("bn1")
        for name, parameter in resnet.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        returned_layers = [1, 2, 3, 4]
        return_layers = {
            f"layer{k}": str(v) for v, k in enumerate(returned_layers)
        }
        self.body = IntermediateLayerGetter(
            resnet, return_layers=return_layers
        )
        self.name = resnet_name

    @property
    def out_channels(self) -> List[int]:
        """Get the number of channels for each level of feature pyramid.

        Raises:
            NotImplementedError: _description_

        Returns:
            List[int]: number of channels
        """
        if self.name in ["resnet18", "resnet34"]:
            return [3, 3] + [64 * 2 ** i for i in range(4)]
        return [3, 3] + [256 * 2 ** i for i in range(4)]

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Torchvision ResNet forward.

        # TODO(tobiasfshr) Add tests

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (List[torch.Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling raio of 2^index.
            fp[0] and fp[1] is a reference to the input images and torchvision
            resnet downsamples the feature maps by 4 directly. The last feature
            map downsamples the input image by 64 with a pooling layer on the
            second last map.
        """
        return [images, images, *self.body(images).values()]
