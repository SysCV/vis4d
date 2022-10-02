"""FCN tests."""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from vis4d.op.base.resnet import ResNet
from vis4d.op.segment.fcn import FCNHead, FCNLoss, FCNOut


class FCN_ResNet(nn.Module):
    def __init__(
        self,
        base_model: str = "res",
        num_classes: int = 21,
        resize: Optional[Tuple[int, int]] = (520, 520),
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """FCN with ResNet, following `torchvision implementation
        <https://github.com/pytorch/vision/blob/torchvision/models/segmentation/
        fcn.py>`_.

        model: FCN_ResNet(base_model="resnet50")
            - dataset: Coco2017
            - recipe: vis4d/model/segment/fcn_resnet_coco_training.py
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
        self.loss = FCNLoss(
            [4, 5],
            nn.CrossEntropyLoss(weights, ignore_index=255),
            weights=[0.5, 1],
        )

    def forward(
        self, images: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Union[Tuple[FCNOut, FCNLoss], FCNOut]:
        features = self.basemodel(images)
        out = self.fcn(features)
        if targets is not None:
            losses = self.loss(out.outputs, targets)
            return out, losses
        return out
