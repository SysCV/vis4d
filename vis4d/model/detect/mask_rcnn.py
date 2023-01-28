"""Mask RCNN model implementation and runtime."""
from __future__ import annotations

import torch
from torch import nn

from vis4d.common import LossesType, ModelOutput
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.box2d import apply_mask, scale_and_clip_boxes
from vis4d.op.box.encoder import BoxEncoder2D
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import (
    Det2Mask,
    MaskRCNNHead,
    MaskRCNNHeadLoss,
    MaskRCNNHeadOut,
    RCNNLoss,
    RoI2Det,
)
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.fpp.fpn import FPN

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^roi_head.mask_head\.", "mask_head."),
    (r"^convs\.", "mask_head.convs."),
    (r"^upsample\.", "mask_head.upsample."),
    (r"^conv_logits\.", "mask_head.conv_logits."),
    (r"^roi_head\.", "faster_rcnn_heads.roi_head."),
    (r"^rpn_head\.", "faster_rcnn_heads.rpn_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class MaskRCNN(nn.Module):
    """Mask RCNN model."""

    def __init__(self, num_classes: int, weights: None | str = None) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of classes.
            weights (None | str, optional): Weights to load for model. If set
                to "mmdet", will load MMDetection pre-trained weights.
                Defaults to None.
        """
        super().__init__()
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            num_classes=num_classes,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        self.mask_head = MaskRCNNHead()
        self.rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_bbox_encoder)
        self.mask_rcnn_loss = MaskRCNNHeadLoss()
        self.transform_outs = RoI2Det(rcnn_bbox_encoder)
        self.det2mask = Det2Mask()

        if weights == "mmdet":
            weights = (
                "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
                "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
                "20200505_003907-3e542a40.pth"
            )
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def forward(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        target_boxes: None | list[torch.Tensor] = None,
        target_classes: None | list[torch.Tensor] = None,
        original_hw: None | list[tuple[int, int]] = None,
    ) -> tuple[FRCNNOut, MaskRCNNHeadOut] | ModelOutput:
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
            tuple[FRCNNOut, MaskRCNNHeadOut] | ModelOutput: Either raw model
                outputs (for training) or predicted outputs (for testing).
        """
        if self.training:
            assert target_boxes is not None and target_classes is not None
            return self.forward_train(
                images, images_hw, target_boxes, target_classes
            )
        assert original_hw is not None
        return self.forward_test(images, images_hw, original_hw)

    def forward_train(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        target_boxes: list[torch.Tensor],
        target_classes: list[torch.Tensor],
    ) -> tuple[FRCNNOut, MaskRCNNHeadOut]:
        """Forward training stage.

        Args:
            images (torch.Tensor): Input images.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            target_boxes (list[torch.Tensor]): Bounding box labels. Required
                for training. Defaults to None.
            target_classes (list[torch.Tensor]): Class labels. Required for
                training. Defaults to None.

        Returns:
            tuple[FRCNNOut, MaskRCNNHeadOut]: Raw model outputs.
        """
        features = self.fpn(self.backbone(images))
        outputs = self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )
        assert outputs.sampled_proposals is not None
        assert outputs.sampled_targets is not None
        pos_proposals = apply_mask(
            [label == 1 for label in outputs.sampled_targets.labels],
            outputs.sampled_proposals.boxes,
        )[0]
        mask_outs = self.mask_head(features, pos_proposals)
        return outputs, mask_outs

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
        boxes, scores, class_ids = self.transform_outs(
            *outs.roi, outs.proposals.boxes, images_hw
        )
        mask_outs = self.mask_head(features, boxes)
        for i, boxs in enumerate(boxes):
            boxes[i] = scale_and_clip_boxes(boxs, original_hw[i], images_hw[i])
        mask_preds = [m.sigmoid() for m in mask_outs.mask_pred]
        masks = self.det2mask(
            mask_preds, boxes, scores, class_ids, original_hw
        )
        return dict(
            boxes2d=boxes,
            boxes2d_scores=scores,
            boxes2d_classes=class_ids,
            masks=masks.masks,
        )


class MaskRCNNLoss(nn.Module):
    """Mask RCNN Loss."""

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        rpn_box_encoder: BoxEncoder2D,
        rcnn_box_encoder: BoxEncoder2D,
    ) -> None:
        """Creates an instance of the class.

        Args:
            anchor_generator (AnchorGenerator): Anchor generator for RPN.
            rpn_box_encoder (BoxEncoder2D): RPN box encoder.
            rcnn_box_encoder (BoxEncoder2D): RCNN box encoder.
        """
        super().__init__()
        self.rpn_loss = RPNLoss(anchor_generator, rpn_box_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_box_encoder)
        self.mask_loss = MaskRCNNHeadLoss()

    def forward(
        self,
        outputs: tuple[FRCNNOut, MaskRCNNHeadOut],
        images_hw: list[tuple[int, int]],
        target_boxes: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> LossesType:
        """Forward of loss function.

        Args:
            outputs (tuple[FRCNNOut, MaskRCNNHeadOut]): Raw model outputs.
            images_hw (list[tuple[int, int]]): Input image resolutions.
            target_boxes (list[torch.Tensor]): Bounding box labels.
            target_masks (list[torch.Tensor]): Instance mask labels.

        Returns:
            LossesType: Dictionary of model losses.
        """
        frcnn_outs, mask_outs = outputs
        rpn_losses = self.rpn_loss(*frcnn_outs.rpn, target_boxes, images_hw)
        assert frcnn_outs.sampled_proposals is not None
        assert frcnn_outs.sampled_targets is not None
        rcnn_losses = self.rcnn_loss(
            *frcnn_outs.roi,
            frcnn_outs.sampled_proposals.boxes,
            frcnn_outs.sampled_targets.labels,
            frcnn_outs.sampled_targets.boxes,
            frcnn_outs.sampled_targets.classes,
        )
        assert frcnn_outs.sampled_target_indices is not None
        sampled_masks = apply_mask(
            frcnn_outs.sampled_target_indices, target_masks
        )[0]
        pos_proposals, pos_classes, pos_mask_targets = apply_mask(
            [label == 1 for label in frcnn_outs.sampled_targets.labels],
            frcnn_outs.sampled_proposals.boxes,
            frcnn_outs.sampled_targets.classes,
            sampled_masks,
        )
        mask_losses = self.mask_loss(
            mask_outs.mask_pred, pos_proposals, pos_classes, pos_mask_targets
        )
        return dict(
            **rpn_losses._asdict(),
            **rcnn_losses._asdict(),
            **mask_losses._asdict(),
        )
