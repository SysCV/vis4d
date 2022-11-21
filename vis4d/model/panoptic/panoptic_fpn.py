"""Mask RCNN tests."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.common import ModelOutput
from vis4d.model.detect.mask_rcnn import REV_KEYS, MaskRCNNLoss
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.box2d import apply_mask, scale_and_clip_boxes
from vis4d.op.detect.faster_rcnn import FasterRCNNHead, FRCNNOut
from vis4d.op.detect.rcnn import (
    Det2Mask,
    MaskRCNNHead,
    MaskRCNNHeadOut,
    RoI2Det,
)
from vis4d.op.fpp.fpn import FPN
from vis4d.op.panoptic.simple_fusion_head import SimplePanopticFusionHead
from vis4d.op.segment.panoptic_fpn_head import (
    PanopticFPNHead,
    PanopticFPNLoss,
    postprocess_segms,
)
from vis4d.op.util import load_model_checkpoint

CNAME = "conv_upsample_layers"
PAN_REV_KEYS = [
    (r"^semantic_head.conv_upsample_layers\.", "conv_upsample_layers."),
    (r"^semantic_head.conv_logits\.", "conv_logits."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
    (rf"^{CNAME}.0.conv.0.weight", f"{CNAME}.0.conv.0.0.weight"),
    (rf"^{CNAME}.0.conv.0.gn\.", f"{CNAME}.0.conv.0.1."),
]
for l in range(4):
    for i in range(l):
        PAN_REV_KEYS += [
            (
                rf"^{CNAME}.{l}.conv.{i}.weight",
                f"{CNAME}.{l}.conv.{i}.0.weight",
            ),
            (rf"^{CNAME}.{l}.conv.{i}.gn\.", f"{CNAME}.{l}.conv.{i}.1."),
        ]


class PanopticFPN(nn.Module):
    """Panoptic FPN model."""

    def __init__(
        self,
        num_things_classes: int,
        num_stuff_classes: int,
        weights: None | str = None,
    ) -> None:
        """Init.

        Args:
            num_things_classes (int): Number of thing (foreground) classes.
            num_stuff_classes (int): Number of stuff (background) classes.
            weights (None | str, optional): Weights to load for model. If
                set to "mmdet", will load MMDetection pre-trained weights.
                Defaults to None.
        """
        super().__init__()
        self.basemodel = ResNet(
            "resnet50", pretrained=True, trainable_layers=3
        )
        self.fpn = FPN(self.basemodel.out_channels[2:], 256)
        self.faster_rcnn = FasterRCNNHead(num_things_classes)
        self.mask_head = MaskRCNNHead(num_things_classes)
        self.seg_head = PanopticFPNHead(num_stuff_classes)
        self.fusion_head = SimplePanopticFusionHead(
            num_things_classes, num_stuff_classes
        )

        self.roi2det = RoI2Det(
            self.faster_rcnn.rcnn_box_encoder, score_threshold=0.5
        )
        self.det2mask = Det2Mask(mask_threshold=0.5)

        self.mask_rcnn_loss = MaskRCNNLoss(
            self.faster_rcnn.anchor_generator,
            self.faster_rcnn.rpn_box_encoder,
            self.faster_rcnn.rcnn_box_encoder,
        )
        self.seg_loss = PanopticFPNLoss(num_things_classes, num_stuff_classes)

        if weights == "mmdet":
            weights = (
                "mmdet://panoptic_fpn/panoptic_fpn_r50_fpn_mstrain_3x_coco/"
                "panoptic_fpn_r50_fpn_mstrain_3x_coco"
                "_20210824_171155-5650f98b.pth"
            )
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
            load_model_checkpoint(self, weights, rev_keys=PAN_REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def forward(
        self,
        images: torch.Tensor,
        input_hw: list[tuple[int, int]],
        target_boxes: None | list[torch.Tensor] = None,
        target_classes: None | list[torch.Tensor] = None,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> tuple[FRCNNOut, MaskRCNNHeadOut, torch.Tensor] | ModelOutput:
        """Forward pass.

        Args:
            images (torch.Tensor): Input images.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            target_boxes (None | list[torch.Tensor], optional): Bounding box
                labels. Required for training. Defaults to None.
            target_classes (None | list[torch.Tensor], optional): Class
                labels. Required for training. Defaults to None.
            original_hw (None | list[tuple[int, int]], optional): Original
                image resolutions (before padding and resizing). Required for
                testing. Defaults to None.

        Returns:
            tuple[FRCNNOut, MaskRCNNHeadOut, torch.Tensor] | ModelOutput:
                Either raw model outputs (for training) or predicted outputs
                (for testing).
        """
        if self.training:
            assert target_boxes is not None and target_classes is not None
            return self.forward_train(
                images, input_hw, target_boxes, target_classes
            )
        assert original_hw is not None
        return self.forward_test(images, input_hw, original_hw)

    def forward_train(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        target_boxes: list[torch.Tensor],
        target_classes: list[torch.Tensor],
    ) -> tuple[FRCNNOut, MaskRCNNHeadOut, torch.Tensor]:
        """Forward training stage.

        Args:
            images (torch.Tensor): Input images.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            target_boxes (list[torch.Tensor]): Bounding box labels. Required
                for training. Defaults to None.
            target_classes (list[torch.Tensor]): Class labels. Required for
                training. Defaults to None.

        Returns:
            tuple[FRCNNOut, MaskRCNNHeadOut, torch.Tensor]: Raw model outputs.
        """
        features = self.fpn(self.basemodel(images))
        outputs = self.faster_rcnn(
            features, images_hw, target_boxes, target_classes
        )
        assert outputs.sampled_proposals is not None
        assert outputs.sampled_targets is not None
        pos_proposals = apply_mask(
            [label == 1 for label in outputs.sampled_targets.labels],
            outputs.sampled_proposals.boxes,
        )[0]
        mask_outs = self.mask_head(features, pos_proposals)
        seg_outs = self.seg_head(features)
        return outputs, mask_outs, seg_outs

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
        features = self.fpn(self.basemodel(images))
        outs = self.faster_rcnn(features, images_hw)
        boxes, scores, class_ids = self.roi2det(
            *outs.roi, outs.proposals.boxes, images_hw
        )
        mask_outs = self.mask_head(features, boxes)
        for j, boxs in enumerate(boxes):
            boxes[j] = scale_and_clip_boxes(boxs, original_hw[j], images_hw[j])
        mask_preds = [m.sigmoid() for m in mask_outs.mask_pred]
        masks = self.det2mask(
            mask_preds, boxes, scores, class_ids, original_hw
        )
        seg_outs = self.seg_head(features)
        post_segs = postprocess_segms(seg_outs, images_hw, original_hw)
        pan_outs = self.fusion_head(masks, post_segs)
        return dict(
            boxes2d=boxes,
            boxes2d_scores=scores,
            boxes2d_classes=class_ids,
            masks=masks.masks,
            pan_mmasks=pan_outs,
        )
