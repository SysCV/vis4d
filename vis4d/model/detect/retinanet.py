"""RetinaNet model implementation and runtime."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.common import LossesType, ModelOutput
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.box2d import scale_and_clip_boxes
from vis4d.op.box.encoder import BoxEncoder2D
from vis4d.op.box.matchers import Matcher
from vis4d.op.box.samplers import Sampler
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.retinanet import (
    Dense2Det,
    RetinaNetHead,
    RetinaNetHeadLoss,
    RetinaNetOut,
)
from vis4d.op.fpp.fpn import FPN, LastLevelP6P7

REV_KEYS = [
    (r"^bbox_head\.", "retinanet_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"^fpn.layer_blocks.3\.", "fpn.extra_blocks.p6_conv."),
    (r"^fpn.layer_blocks.4\.", "fpn.extra_blocks.p7_conv."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class RetinaNet(nn.Module):
    """RetinaNet wrapper class for checkpointing etc."""

    def __init__(self, num_classes: int, weights: None | str = None) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of classes.
            weights (None | str, optional): Weights to load for model. If
                set to "mmdet", will load MMDetection pre-trained weights.
                Defaults to None.
        """
        super().__init__()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(
            self.backbone.out_channels[3:],
            256,
            LastLevelP6P7(2048, 256),
            start_index=3,
        )
        self.retinanet_head = RetinaNetHead(
            num_classes=num_classes, in_channels=256
        )
        self.transform_outs = Dense2Det(
            self.retinanet_head.anchor_generator,
            self.retinanet_head.box_encoder,
            num_pre_nms=1000,
            max_per_img=100,
            nms_threshold=0.5,
            score_thr=0.05,
        )

        if weights == "mmdet":
            weights = (
                "mmdet://retinanet/retinanet_r50_fpn_2x_coco/"
                "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
            )
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def forward(
        self,
        images: torch.Tensor,
        input_hw: None | list[tuple[int, int]] = None,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> RetinaNetOut | ModelOutput:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images.
            input_hw (None | list[tuple[int, int]], optional): Input image
                resolutions. Defaults to None.
            original_hw (None | list[tuple[int, int]], optional): Original
                image resolutions (before padding and resizing). Required for
                testing. Defaults to None.

        Returns:
            RetinaNetOut | ModelOutput: Either raw model outputs (for
                training) or predicted outputs (for testing).
        """
        if self.training:
            return self.forward_train(images)
        assert input_hw is not None and original_hw is not None
        return self.forward_test(images, input_hw, original_hw)

    def forward_train(self, images: torch.Tensor) -> RetinaNetOut:
        """Forward training stage.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            RetinaNetOut: Raw model outputs.
        """
        features = self.fpn(self.backbone(images))
        return self.retinanet_head(features[-5:])

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
        outs = self.retinanet_head(features[-5:])
        boxes, scores, class_ids = self.transform_outs(
            cls_outs=outs.cls_score,
            reg_outs=outs.bbox_pred,
            images_hw=images_hw,
        )
        for i, boxs in enumerate(boxes):
            boxes[i] = scale_and_clip_boxes(boxs, original_hw[i], images_hw[i])
        return {
            "boxes2d": boxes,
            "boxes2d_scores": scores,
            "boxes2d_classes": class_ids,
        }


class RetinaNetLoss(nn.Module):
    """RetinaNet Loss."""

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: BoxEncoder2D,
        box_matcher: Matcher,
        box_sampler: Sampler,
    ) -> None:
        """Creates an instance of the class.

        Args:
            anchor_generator (AnchorGenerator): Anchor generator for RPN.
            box_encoder (BoxEncoder2D): Bounding box encoder.
            box_matcher (BaseMatcher): Bounding box matcher.
            box_sampler (BaseSampler): Bounding box sampler.
        """
        super().__init__()
        self.retinanet_loss = RetinaNetHeadLoss(
            anchor_generator, box_encoder, box_matcher, box_sampler
        )

    def forward(
        self,
        outputs: RetinaNetOut,
        images_hw: list[tuple[int, int]],
        target_boxes: list[torch.Tensor],
        target_classes: list[torch.Tensor],
    ) -> LossesType:
        """Forward of loss function.

        Args:
            outputs (RetinaNetOut): Raw model outputs.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            target_boxes (list[torch.Tensor]): Bounding box labels.
            target_classes (list[torch.Tensor]): Class labels.

        Returns:
            LossesType: Dictionary of model losses.
        """
        losses = self.retinanet_loss(
            outputs.cls_score,
            outputs.bbox_pred,
            target_boxes,
            images_hw,
            target_classes,
        )
        return losses._asdict()
