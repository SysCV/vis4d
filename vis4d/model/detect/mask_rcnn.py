"""Mask RCNN model implementation and runtime."""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.op.base.resnet import ResNet
from vis4d.op.box.box2d import apply_mask, bbox_postprocess
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
    DetOut,
    MaskOut,
    MaskRCNNHead,
    MaskRCNNHeadLoss,
    RCNNLoss,
    RoI2Det,
)
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.struct_to_revise import LossesType, ModelOutput

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

    def __init__(
        self, num_classes: int, weights: Optional[str] = None
    ) -> None:
        """Init."""
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
        self.mask_head = MaskRCNNHead()
        self.rpn_loss = RPNLoss(self.anchor_gen, self.rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(self.rcnn_bbox_encoder)
        self.mask_rcnn_loss = MaskRCNNHeadLoss()
        self.transform_outs = RoI2Det(self.rcnn_bbox_encoder)
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
        images_hw: List[Tuple[int, int]],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        original_hw: Optional[List[Tuple[int, int]]] = None,
    ) -> Union[Tuple[FRCNNOut, MaskOut], ModelOutput]:
        """Forward."""
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
        images_hw: List[Tuple[int, int]],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> Tuple[FRCNNOut, MaskOut]:
        """Forward training stage."""
        features = self.fpn(self.backbone(images))
        outputs = self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )
        pos_proposals = apply_mask(
            [label == 1 for label in outputs.sampled_targets.labels],
            outputs.sampled_proposals.boxes,
        )[0]
        mask_outs = self.mask_head(features[2:-1], pos_proposals)
        return outputs, mask_outs

    def forward_test(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        original_hw: List[Tuple[int, int]],
    ) -> ModelOutput:
        """Forward testing stage."""
        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        boxes, scores, class_ids = self.transform_outs(
            *outs.roi, outs.proposals.boxes, images_hw
        )
        mask_outs = self.mask_head(features[2:-1], boxes)
        for i, boxs in enumerate(boxes):
            boxes[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        post_dets = DetOut(boxes=boxes, scores=scores, class_ids=class_ids)
        masks = self.det2mask(
            mask_outs=mask_outs.mask_pred.sigmoid(),
            dets=post_dets,
            images_hw=original_hw,
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
        """Init."""
        super().__init__()
        self.rpn_loss = RPNLoss(anchor_generator, rpn_box_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_box_encoder)
        self.mask_loss = MaskRCNNHeadLoss()

    def forward(
        self,
        outputs: Tuple[FRCNNOut, MaskOut],
        images_hw: List[Tuple[int, int]],
        target_boxes: List[torch.Tensor],
        target_masks: List[torch.Tensor],
    ) -> LossesType:
        """Forward."""
        frcnn_outs, mask_outs = outputs
        rpn_losses = self.rpn_loss(*frcnn_outs.rpn, target_boxes, images_hw)
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
