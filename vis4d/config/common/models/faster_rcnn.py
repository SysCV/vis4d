"""Faseter R-CNN base model config."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference

from vis4d.config import class_config
from vis4d.engine.connectors import LossConnector, data_key, pred_key
from vis4d.engine.loss_module import LossModule
from vis4d.model.detect.faster_rcnn import FasterRCNN
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder, DeltaXYWHBBoxEncoder
from vis4d.op.box.matchers import MaxIoUMatcher
from vis4d.op.box.samplers import RandomSampler
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RCNNHead, RCNNLoss
from vis4d.op.detect.rpn import RPNLoss

# Data connectors
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
}


def get_default_rpn_box_codec_cfg(
    target_means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    target_stds: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[ConfigDict, ConfigDict]:
    """Get default config for rpn box encoder and decoder."""
    return tuple(
        class_config(x, target_means=target_means, target_stds=target_stds)
        for x in (DeltaXYWHBBoxEncoder, DeltaXYWHBBoxDecoder)
    )


def get_default_rcnn_box_codec_cfg(
    target_means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    target_stds: tuple[float, ...] = (0.1, 0.1, 0.2, 0.2),
) -> tuple[ConfigDict, ConfigDict]:
    """Get default config for rcnn box encoder and decoder."""
    return tuple(
        class_config(x, target_means=target_means, target_stds=target_stds)
        for x in (DeltaXYWHBBoxEncoder, DeltaXYWHBBoxDecoder)
    )


def get_faster_rcnn_cfg(
    num_classes: FieldReference | int,
    basemodel: ConfigDict,
    weights: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Return default config for faster_rcnn model and loss.

    This is an example for setting every component of the model and loss.
    Everything is the same as the default args.

    Args:
        num_classes (FieldReference | int): Number of classes.
        basemodel (ConfigDict): Base model config.
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
        FasterRCNN,
        num_classes=num_classes,
        basemodel=basemodel,
        faster_rcnn_head=faster_rcnn_head,
        rcnn_box_decoder=rcnn_box_decoder,
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
        ],
    )
    return model, loss
