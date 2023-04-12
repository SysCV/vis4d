"""Loss for faster_rcnn model."""
from __future__ import annotations

from torch import nn

from vis4d.engine.loss import WeightedMultiLoss
from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.rcnn import (
    MaskRCNNHeadLoss,
    RCNNLoss,
    SampledMaskLoss,
    positive_mask_sampler,
)
from vis4d.op.detect.rpn import RPNLoss


def get_default_mask_rcnn_loss(
    anchor_generator: AnchorGenerator,
    rpn_box_encoder: DeltaXYWHBBoxEncoder,
    rcnn_box_encoder: DeltaXYWHBBoxEncoder,
) -> nn.Module:
    """Return default loss for faster_rcnn model.

    This loss consists of a RPN loss as well as a RCNN and Mask loss.

    Returns:
        nn.Module: Loss module.
    """
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
