"""Loss for faster_rcnn model."""
from __future__ import annotations

from torch import nn

from vis4d.engine.loss import WeightedMultiLoss
from vis4d.op.box.encoder.base import BoxEncoder2D
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss


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
