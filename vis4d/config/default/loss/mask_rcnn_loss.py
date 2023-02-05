"""Loss for faster_rcnn model."""
from __future__ import annotations

import torch.nn as nn

from vis4d.engine.loss import WeightedMultiLoss
from vis4d.op.detect.faster_rcnn import (
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import (
    MaskRCNNHeadLoss,
    RCNNLoss,
    SampledMaskLoss,
    positive_mask_sampler,
)
from vis4d.op.detect.rpn import RPNLoss


def get_default_mask_rcnn_loss() -> nn.Module:
    """Return default loss for faster_rcnn model.

    This loss consists of a RPN loss as well as a RCNN loss.
    # TODO: Add better docstring
    # TODO make generators shared with model.

    Returns:
        nn.Module: Loss module.
    """
    anchor_generator = get_default_anchor_generator()
    rpn_box_encoder = get_default_rpn_box_encoder()
    rcnn_box_encoder = get_default_rcnn_box_encoder()

    rpn_loss = RPNLoss(anchor_generator, rpn_box_encoder)
    rcnn_loss = RCNNLoss(rcnn_box_encoder)

    mask_loss = SampledMaskLoss(positive_mask_sampler, MaskRCNNHeadLoss())

    return WeightedMultiLoss(
        [
            {"loss": rpn_loss, "weight": 1.0},
            {"loss": rcnn_loss, "weight": 1.0},
            {"loss": mask_loss, "weight": 1.0},
        ]
    )
