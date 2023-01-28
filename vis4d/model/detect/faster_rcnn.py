"""Faster RCNN model implementation and runtime."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.common import ModelOutput
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.box2d import scale_and_clip_boxes
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import RoI2Det
from vis4d.op.fpp.fpn import FPN

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "faster_rcnn_heads.rpn_head.rpn_box."),
    (r"^rpn_head.rpn_", "faster_rcnn_heads.rpn_head.rpn_"),
    (r"^roi_head.bbox_head\.", "faster_rcnn_heads.roi_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class FasterRCNN(nn.Module):
    """Faster RCNN model."""

    def __init__(self, num_classes: int, weights: None | str = None) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of classes.
            weights (None | str, optional): Weights to load for model. If
                set to "mmdet", will load MMDetection pre-trained weights.
                Defaults to None.
        """
        super().__init__()
        self.anchor_gen = get_default_anchor_generator()
        self.rpn_bbox_encoder = get_default_rpn_box_encoder()
        self.rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            num_classes=num_classes,
            anchor_generator=self.anchor_gen,
            rpn_box_encoder=self.rpn_bbox_encoder,
            rcnn_box_encoder=self.rcnn_bbox_encoder,
        )
        self.roi2det = RoI2Det(self.rcnn_bbox_encoder)

        if weights == "mmdet":
            weights = (
                "mmdet://faster_rcnn/faster_rcnn_r50_fpn_1x_coco/"
                "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
            )
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def forward(
        self,
        images: torch.Tensor,
        input_hw: list[tuple[int, int]],
        boxes2d: None | list[torch.Tensor] = None,
        boxes2d_classes: None | list[torch.Tensor] = None,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> FRCNNOut | ModelOutput:
        # TODO, why do we have both.
        #  Why not just model output with different
        #  keys for training outs e.g. class_logits?
        """Forward pass.

        Args:
            images (torch.Tensor): Input images.
            input_hw (list[tuple[int, int]]): Input image resolutions.
            boxes2d (None | list[torch.Tensor], optional): Bounding box labels.
                Required for training. Defaults to None.
            boxes2d_classes (None | list[torch.Tensor], optional): Class
                labels. Required for training. Defaults to None.
            original_hw (None | list[tuple[int, int]], optional): Original
                image resolutions (before padding and resizing). Required for
                testing. Defaults to None.

        Returns:
            FRCNNOut | ModelOutput: Either raw model outputs (for
                training) or predicted outputs (for testing).
        """
        if self.training:
            assert boxes2d is not None and boxes2d_classes is not None
            return self.forward_train(
                images, input_hw, boxes2d, boxes2d_classes
            )
        assert original_hw is not None
        return self.forward_test(images, input_hw, original_hw)

    def __call__(
        self,
        images: torch.Tensor,
        input_hw: list[tuple[int, int]],
        boxes2d: None | list[torch.Tensor] = None,
        boxes2d_classes: None | list[torch.Tensor] = None,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> FRCNNOut | ModelOutput:
        """Type definition for call implementation."""
        return self._call_impl(
            images, input_hw, boxes2d, boxes2d_classes, original_hw
        )

    def forward_train(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        target_boxes: list[torch.Tensor],
        target_classes: list[torch.Tensor],
    ) -> FRCNNOut:
        """Forward training stage.

        Args:
            images (torch.Tensor): Input images.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            target_boxes (list[torch.Tensor]): Bounding box labels.
            target_classes (list[torch.Tensor]): Class labels.

        Returns:
            FRCNNOut: Raw model outputs.
        """
        features = self.fpn(self.backbone(images))
        return self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )

    def forward_test(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
    ) -> ModelOutput:
        """Forward testing stage.

        Args:
            images (torch.Tensor): Input images.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            original_hw (list[tuple[int, int]]): Original image resolutions
                (before padding and resizing).

        Returns:
            ModelOutput: Predicted outputs.
        """
        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        boxes, scores, class_ids = self.roi2det(
            *outs.roi, outs.proposals.boxes, images_hw
        )

        for i, boxs in enumerate(boxes):
            boxes[i] = scale_and_clip_boxes(boxs, original_hw[i], images_hw[i])
        return dict(
            boxes2d=boxes, boxes2d_scores=scores, boxes2d_classes=class_ids
        )
