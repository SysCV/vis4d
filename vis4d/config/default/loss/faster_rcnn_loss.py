"""Loss for faster_rcnn model."""
from __future__ import annotations

import torch.nn as nn

from vis4d.engine.loss import WeightedMultiLoss
from vis4d.op.detect.faster_rcnn import (
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss


def get_default_faster_rcnn_loss() -> nn.Module:
    """Return default loss for faster_rcnn model.

    This los consists of a RPN loss as well as a RCNN loss.
    # TODO: Add better docstring

    Returns:
        nn.Module: Loss module.
    """
    anchor_generator = get_default_anchor_generator()
    rpn_box_encoder = get_default_rpn_box_encoder()
    rcnn_box_encoder = get_default_rcnn_box_encoder()
    rpn_loss = RPNLoss(anchor_generator, rpn_box_encoder)
    rcnn_loss = RCNNLoss(rcnn_box_encoder)

    return WeightedMultiLoss(
        [{"loss": rpn_loss, "weight": 1.0}, {"loss": rcnn_loss, "weight": 1.0}]
    )


# print(get_default_faster_rcnn_loss())
