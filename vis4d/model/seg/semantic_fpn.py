"""SemanticFPN Implementation."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base import BaseModel, ResNetV1c
from vis4d.op.fpp.fpn import FPN
from vis4d.op.mask.util import clip_mask
from vis4d.op.seg.semantic_fpn import SemanticFPNHead, SemanticFPNOut

REV_KEYS = [
    (r"^decode_head\.", "seg_head."),
    (r"^classifier\.", "fcn.heads.1."),
    (r"^backbone\.", "basemodel."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]
for ki in range(4):
    for kj in range(5):
        REV_KEYS += [
            (
                rf"^seg_head.scale_heads\.{ki}\.{kj}\.bn\.",
                f"seg_head.scale_heads.{ki}.{kj}.norm.",
            )
        ]


class MaskOut(NamedTuple):
    """Output mask predictions."""

    masks: list[torch.Tensor]  # list of masks for each image


class SemanticFPN(nn.Module):
    """Semantic FPN.

    Args:
        num_classes (int): Number of classes.
        resize (bool): Resize output to input size.
        weights (None | str): Pre-trained weights.
        basemodel (None | BaseModel): Base model to use. If None is passed,
            this will default to ResNetV1c
    """

    def __init__(
        self,
        num_classes: int,
        resize: bool = True,
        weights: None | str = None,
        basemodel: None | BaseModel = None,
    ):
        """Init."""
        super().__init__()
        self.resize = resize
        if basemodel is None:
            basemodel = ResNetV1c(
                "resnet50_v1c",
                pretrained=True,
                trainable_layers=5,
                norm_frozen=False,
            )

        self.basemodel = basemodel
        self.fpn = FPN(self.basemodel.out_channels[2:], 256, extra_blocks=None)
        self.seg_head = SemanticFPNHead(num_classes, 256)

        if weights is not None:
            if weights.startswith("mmseg://") or weights.startswith(
                "bdd100k://"
            ):
                load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
            else:
                load_model_checkpoint(self, weights)

    def forward_train(self, images: torch.Tensor) -> SemanticFPNOut:
        """Forward pass for training.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            SemanticFPNOut: Raw model predictions.
        """
        features = self.fpn(self.basemodel(images.contiguous()))
        out = self.seg_head(features)
        if self.resize:
            return SemanticFPNOut(
                outputs=F.interpolate(
                    out.outputs,
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return out

    def forward_test(
        self, images: torch.Tensor, original_hw: list[tuple[int, int]]
    ) -> MaskOut:
        """Forward pass for testing.

        Args:
            images (torch.Tensor): Input images.
            original_hw (list[tuple[int, int]], optional): Original image
                resolutions (before padding and resizing). Required for
                testing.

        Returns:
            SemanticFPNOut: Raw model predictions.
        """
        features = self.fpn(self.basemodel(images))
        out = self.seg_head(features)

        new_masks = []
        for i, outputs in enumerate(out.outputs):
            opt = F.interpolate(
                outputs.unsqueeze(0),
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            new_masks.append(clip_mask(opt, original_hw[i]).argmax(dim=0))
        return MaskOut(masks=new_masks)

    def forward(
        self,
        images: torch.Tensor,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> SemanticFPNOut | MaskOut:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images.
            original_hw (None | list[tuple[int, int]], optional): Original
                image resolutions (before padding and resizing). Required for
                testing. Defaults to None.

        Returns:
            MaskOut: Raw model predictions.
        """
        if self.training:
            return self.forward_train(images)
        assert original_hw is not None
        return self.forward_test(images, original_hw)
