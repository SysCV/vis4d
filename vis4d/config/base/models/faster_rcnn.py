"""Faseter R-CNN base model config."""
from __future__ import annotations

from torch import nn

from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key
from vis4d.engine.loss import WeightedMultiLoss
from vis4d.op.box.encoder.base import BoxEncoder2D
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss

# Data connectors
CONN_BBOX_2D_TRAIN = {
    K.images: K.images,
    K.input_hw: K.input_hw,
    K.boxes2d: K.boxes2d,
    K.boxes2d_classes: K.boxes2d_classes,
}

CONN_BBOX_2D_TEST = {
    **CONN_BBOX_2D_TRAIN,
    "original_hw": "original_hw",
}

CONN_RPN_LOSS_2D = {
    "cls_outs": pred_key("rpn.cls"),
    "reg_outs": pred_key("rpn.box"),
    "target_boxes": data_key("boxes2d"),
    "images_hw": data_key("input_hw"),
}

CONN_ROI_LOSS_2D = {
    "class_outs": pred_key("roi.cls_score"),
    "regression_outs": pred_key("roi.bbox_pred"),
    "boxes": pred_key("sampled_proposals.boxes"),
    "boxes_mask": pred_key("sampled_targets.labels"),
    "target_boxes": pred_key("sampled_targets.boxes"),
    "target_classes": pred_key("sampled_targets.classes"),
    "pred_sampled_proposals": pred_key("sampled_proposals"),
}


# Loss config
def get_default_faster_rcnn_loss(
    anchor_generator: AnchorGenerator,
    rpn_box_encoder: BoxEncoder2D,
    rcnn_box_encoder: BoxEncoder2D,
) -> nn.Module:
    """Return default loss for faster_rcnn model.

    This los consists of a RPN loss as well as a RCNN loss.
    # TODO: Add better docstring

    Returns:
        nn.Module: Loss module.
    """
    rpn_loss = RPNLoss(anchor_generator, rpn_box_encoder)
    rcnn_loss = RCNNLoss(rcnn_box_encoder)

    return WeightedMultiLoss(
        [{"loss": rpn_loss, "weight": 1.0}, {"loss": rcnn_loss, "weight": 1.0}]
    )
