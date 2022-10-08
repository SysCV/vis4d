"""Faster RCNN model implementation and runtime."""
from typing import List, Optional, Union

import torch
from torch import nn

from vis4d.data.datasets.base import COMMON_KEYS, DictData
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.util import bbox_postprocess
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
from vis4d.struct_to_revise import LossesType, ModelOutput

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^backbone\.", "body."),
    (r"^neck.lateral_convs\.", "inner_blocks."),
    (r"^neck.fpn_convs\.", "layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class FasterRCNN(nn.Module):
    """Faster RCNN model."""

    def __init__(
        self, num_classes: int, weights: Optional[str] = None
    ) -> None:
        """Init."""
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
        self.rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_bbox_encoder)
        self.transform_outs = RoI2Det(rcnn_bbox_encoder)

        if weights == "mmdet":
            weights = (
                "mmdet://faster_rcnn/faster_rcnn_r50_fpn_1x_coco/"
                "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
            )
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)
        elif weights is not None:
            load_model_checkpoint(self, weights)

    def visualize_proposals(
        self, images: torch.Tensor, outs: FRCNNOut, topk: int = 100
    ) -> None:
        """Visualize topk proposals."""
        from vis4d.vis.image import imshow_bboxes

        for im, boxes, scores in zip(images, *outs.proposals):
            _, topk_indices = torch.topk(scores, topk)
            imshow_bboxes(im, boxes[topk_indices])

    def forward(self, data: DictData) -> Union[LossesType, ModelOutput]:
        """Forward."""
        if self.training:
            return self._forward_train(data)
        return self._forward_test(data)

    def _forward_train(self, data: DictData) -> LossesType:
        """Forward training stage."""
        images, images_hw, target_boxes, target_classes = (
            data[COMMON_KEYS.images],
            data[COMMON_KEYS.metadata]["input_hw"],
            data[COMMON_KEYS.boxes2d],
            data[COMMON_KEYS.boxes2d_classes],
        )

        features = self.fpn(self.backbone(images))
        outputs = self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )

        rpn_losses = self.rpn_loss(*outputs.rpn, target_boxes, images_hw)
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.labels,
            outputs.sampled_targets.boxes,
            outputs.sampled_targets.classes,
        )
        return dict(**rpn_losses._asdict(), **rcnn_losses._asdict())

    def _forward_test(self, data: DictData) -> ModelOutput:
        """Forward testing stage."""
        images = data[COMMON_KEYS.images]
        original_hw = data[COMMON_KEYS.metadata]["original_hw"]
        images_hw = data[COMMON_KEYS.metadata]["input_hw"]

        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        boxes, scores, class_ids = self.transform_outs(
            *outs.roi, outs.proposals.boxes, images_hw
        )

        for i, boxs in enumerate(boxes):
            boxes[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        output = dict(
            boxes2d=boxes, boxes2d_scores=scores, boxes2d_classes=class_ids
        )
        return output
