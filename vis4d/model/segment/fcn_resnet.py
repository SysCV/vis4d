"""FCN tests."""
from typing import Optional, Tuple

import torch
from torch import nn

from vis4d.op.base.resnet import ResNet
from vis4d.op.segment.fcn import FCNHead, FCNLoss, FCNOut


REV_KEYS = [
    (r"^backbone\.", "basemodel.body."),
    (r"^aux_classifier\.", "fcn.heads.0."),
    (r"^classifier\.", "fcn.heads.1."),
]


class FCNResNet(nn.Module):
    def __init__(
        self,
        base_model: str = "resnet50",
        num_classes: int = 21,
        resize: Optional[Tuple[int, int]] = (520, 520),
    ) -> None:
        """FCN with ResNet, following `torchvision implementation
        <https://github.com/pytorch/vision/blob/torchvision/models/segmentation/
        fcn.py>`_.

        model: FCNResNet(base_model="resnet50")
            - dataset: Coco2017
            - recipe: vis4d/model/segment/FCNResNet_coco_training.py
            - metrics:
                - mIoU: 62.52
                - Acc: 90.50
        """
        super().__init__()
        if base_model.startswith("resnet"):
            self.basemodel = ResNet(
                base_model,
                pretrained=True,
                replace_stride_with_dilation=[False, True, True],
            )
        else:
            raise ValueError("base model not supported!")
        self.fcn = FCNHead(
            self.basemodel.out_channels[4:], num_classes, resize=resize
        )
        print(self.basemodel.out_channels[4:])

    def forward(self, images: torch.Tensor) -> FCNOut:
        features = self.basemodel(images)
        out = self.fcn(features)
        return out


class FCNResNetLoss(nn.Module):
    """FCNResNet Loss."""

    def __init__(self, weights: Optional[torch.Tensor] = None) -> None:
        """Init."""
        super().__init__()
        self.loss = FCNLoss(
            [4, 5],
            nn.CrossEntropyLoss(weights, ignore_index=255),
            weights=[0.5, 1],
        )

    def forward(self, out: FCNOut, targets: torch.Tensor) -> FCNLoss:
        """Forward of loss function.

        Args:
            out (FCNOut): Raw model outputs.
            targets (torch.Tensor): Segmentation masks
        Returns:
            FCNLoss: Dictionary of model losses.
        """
        losses = self.loss(out.outputs, targets)
        return losses
