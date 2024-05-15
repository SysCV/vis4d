"""Mask RCNN base model config."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference

from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import (
    LossConnector,
    data_key,
    pred_key,
    remap_pred_keys,
)
from vis4d.engine.loss_module import LossModule
from vis4d.model.detect.mask_rcnn import MaskRCNN
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.matchers import MaxIoUMatcher
from vis4d.op.box.samplers import RandomSampler
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.mask_rcnn import (
    MaskRCNNHead,
    MaskRCNNHeadLoss,
    SampledMaskLoss,
    positive_mask_sampler,
)
from vis4d.op.detect.rcnn import RCNNHead, RCNNLoss
from vis4d.op.detect.rpn import RPNLoss
from vis4d.zoo.base import get_callable_cfg
from vis4d.zoo.base.models.faster_rcnn import (
    CONN_ROI_LOSS_2D as _CONN_ROI_LOSS_2D,
)
from vis4d.zoo.base.models.faster_rcnn import (
    CONN_RPN_LOSS_2D as _CONN_RPN_LOSS_2D,
)
from vis4d.zoo.base.models.faster_rcnn import (
    get_default_rcnn_box_codec_cfg,
    get_default_rpn_box_codec_cfg,
)

# Data connectors
CONN_MASK_HEAD_LOSS_2D = {
    "mask_preds": pred_key("masks.mask_pred"),
    "target_masks": data_key(K.instance_masks),
    "sampled_target_indices": pred_key("boxes.sampled_target_indices"),
    "sampled_targets": pred_key("boxes.sampled_targets"),
    "sampled_proposals": pred_key("boxes.sampled_proposals"),
}

CONN_RPN_LOSS_2D = remap_pred_keys(_CONN_RPN_LOSS_2D, "boxes")

CONN_ROI_LOSS_2D = remap_pred_keys(_CONN_ROI_LOSS_2D, "boxes")


def get_mask_rcnn_cfg(
    num_classes: FieldReference | int,
    basemodel: ConfigDict,
    no_overlap: bool = False,
    weights: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Return default config for mask_rcnn model and loss.

    This is an example for setting every component of the model and loss.
    Everything is the same as the default args.

    Args:
        num_classes (FieldReference | int): Number of classes.
        basemodel (ConfigDict): Base model config.
        no_overlap (bool, optional): Whether to remove overlapping pixels
            between masks. Defaults to False.
        weights (str | None, optional): Weights to load. Defaults to None.
    """
    ######################################################
    ##                        MODEL                     ##
    ######################################################
    anchor_generator = class_config(
        AnchorGenerator,
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
    )

    rpn_box_encoder, rpn_box_decoder = get_default_rpn_box_codec_cfg()
    rcnn_box_encoder, rcnn_box_decoder = get_default_rcnn_box_codec_cfg()

    box_matcher = class_config(
        MaxIoUMatcher,
        thresholds=[0.5],
        labels=[0, 1],
        allow_low_quality_matches=False,
    )

    box_sampler = class_config(
        RandomSampler, batch_size=512, positive_fraction=0.25
    )

    roi_head = class_config(RCNNHead, num_classes=num_classes)

    mask_head = class_config(MaskRCNNHead, num_classes=num_classes)

    faster_rcnn_head = class_config(
        FasterRCNNHead,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        rpn_box_decoder=rpn_box_decoder,
        box_matcher=box_matcher,
        box_sampler=box_sampler,
        roi_head=roi_head,
    )

    model = class_config(
        MaskRCNN,
        num_classes=num_classes,
        basemodel=basemodel,
        faster_rcnn_head=faster_rcnn_head,
        mask_head=mask_head,
        rcnn_box_decoder=rcnn_box_decoder,
        no_overlap=no_overlap,
        weights=weights,
    )

    ######################################################
    ##                      LOSS                        ##
    ######################################################
    rpn_loss = class_config(
        RPNLoss,
        anchor_generator=anchor_generator,
        box_encoder=rpn_box_encoder,
    )
    rcnn_loss = class_config(
        RCNNLoss, box_encoder=rcnn_box_encoder, num_classes=num_classes
    )

    mask_loss = class_config(
        SampledMaskLoss,
        mask_sampler=get_callable_cfg(positive_mask_sampler),
        loss=class_config(MaskRCNNHeadLoss, num_classes=num_classes),
    )

    loss = class_config(
        LossModule,
        losses=[
            {
                "loss": rpn_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_RPN_LOSS_2D
                ),
            },
            {
                "loss": rcnn_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_ROI_LOSS_2D
                ),
            },
            {
                "loss": mask_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_MASK_HEAD_LOSS_2D
                ),
            },
        ],
    )
    return model, loss
