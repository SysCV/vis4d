"""SemanticFPN Implementation."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.typing import LossesType
from vis4d.op.base import BaseModel, ResNet
from vis4d.op.fpp.fpn import FPN
from vis4d.op.segment.loss import SegmentLoss
from vis4d.op.segment.semantic_fpn import SemanticFPNHead, SemanticFPNOut

REV_KEYS = [
    (r"^decode_head\.", "seg_head."),
    (r"\.bn\.weight", ".1.weight"),
    (r"\.bn\.bias", ".1.bias"),
    (r"\.bn\.running_mean", ".1.running_mean"),
    (r"\.bn\.running_var", ".1.running_var"),
    (r"^classifier\.", "fcn.heads.1."),
    (r"^backbone\.", "basemodel.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class SemanticFPN(nn.Module):
    """Semantic FPN."""

    def __init__(
        self,
        num_classes: int,
        weights: None | str = None,
        basemodel: BaseModel = ResNet(
            "resnet50", pretrained=True, trainable_layers=3
        ),
    ) -> None:
        """Semantic FPN."""
        super().__init__()
        self.basemodel = basemodel
        self.fpn = FPN(self.basemodel.out_channels[2:], 256)
        self.seg_head = SemanticFPNHead(num_classes, 256)

        if weights == "mmseg":
            weights = (
                "mmseg://sem_fpn/fpn_r50_512x1024_80k_cityscapes/"
                "fpn_r50_512x1024_80k_cityscapes_20200717_021437-94018a0d.pth"
            )
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
        elif weights == "bdd100k":
            # TODO: add bdd100k pre-trained weights
            pass
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def forward_train(self, images: torch.Tensor) -> SemanticFPNOut:
        """Forward pass for training.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            SemanticFPNOut: Raw model predictions.
        """
        return self.forward(images)

    def forward_test(self, images: torch.Tensor) -> SemanticFPNOut:
        """Forward pass for testing.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            SemanticFPNOut: Raw model predictions.
        """
        return self.forward(images)

    def forward(self, images: torch.Tensor) -> SemanticFPNOut:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            SemanticFPNOut: Raw model predictions.
        """
        features = self.fpn(self.basemodel(images))
        out = self.seg_head(features)
        return out


class SemanticFPNLoss(nn.Module):
    """SemanticFPN Loss."""

    def __init__(self, weights: None | torch.Tensor = None) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.loss = SegmentLoss(
            loss_fn=nn.CrossEntropyLoss(weights, ignore_index=255)
        )

    def forward(self, outs: torch.Tensor, targets: torch.Tensor) -> LossesType:
        """Forward of loss function.

        Args:
            outs (torch.Tensor): Raw model outputs.
            targets (torch.Tensor): Segmentation masks.

        Returns:
            LossesType: Dictionary of model losses.
        """
        losses = self.loss([outs], targets.long())
        return losses
