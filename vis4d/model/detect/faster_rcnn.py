"""Faster RCNN model implementation and runtime."""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.common import LossesType, ModelOutput
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.box2d import bbox_postprocess
from vis4d.op.box.encoder import BoxEncoder2D
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import RCNNLoss, RoI2Det
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.vis.image import imshow_bboxes

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^backbone\.", "body."),
    (r"^neck.lateral_convs\.", "inner_blocks."),
    (r"^neck.fpn_convs\.", "layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


def visualize_proposals(
    images: torch.Tensor, outs: FRCNNOut, topk: int = 100
) -> None:
    """Visualize topk proposals."""
    for im, boxes, scores in zip(images, *outs.proposals):
        _, topk_indices = torch.topk(scores, topk)
        imshow_bboxes(im, boxes[topk_indices])


class FasterRCNN(nn.Module):
    """Faster RCNN model."""

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
        self.transform_outs = RoI2Det(self.rcnn_bbox_encoder)

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
        input_hw: List[Tuple[int, int]],
        boxes2d: Optional[List[torch.Tensor]] = None,
        boxes2d_classes: Optional[List[torch.Tensor]] = None,
        original_hw: Optional[List[Tuple[int, int]]] = None,
    ) -> Union[FRCNNOut, ModelOutput]:
        """Forward."""
        if self.training:
            assert boxes2d is not None and boxes2d_classes is not None
            return self.forward_train(
                images, input_hw, boxes2d, boxes2d_classes
            )
        assert original_hw is not None
        return self.forward_test(images, input_hw, original_hw)

    def forward_train(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> FRCNNOut:
        """Forward training stage."""
        features = self.fpn(self.backbone(images))
        return self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )

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

        for i, boxs in enumerate(boxes):
            boxes[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        return dict(
            boxes2d=boxes, boxes2d_scores=scores, boxes2d_classes=class_ids
        )


class FasterRCNNLoss(nn.Module):
    """Faster RCNN Loss."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        anchor_generator = get_default_anchor_generator()
        rpn_box_encoder = get_default_rpn_box_encoder()
        rcnn_box_encoder = get_default_rcnn_box_encoder()
        self.rpn_loss = RPNLoss(anchor_generator, rpn_box_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_box_encoder)

    def forward(
        self,
        outputs: FRCNNOut,
        input_hw: List[Tuple[int, int]],
        boxes2d: List[torch.Tensor],
    ) -> LossesType:
        """Forward."""
        rpn_losses = self.rpn_loss(*outputs.rpn, boxes2d, input_hw)
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.labels,
            outputs.sampled_targets.boxes,
            outputs.sampled_targets.classes,
        )
        return dict(**rpn_losses._asdict(), **rcnn_losses._asdict())
