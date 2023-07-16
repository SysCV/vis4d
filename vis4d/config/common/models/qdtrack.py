"""QD-Track model config."""
from __future__ import annotations

from ml_collections import ConfigDict, FieldReference

from vis4d.config import class_config
from vis4d.config.common.models.faster_rcnn import (
    CONN_ROI_LOSS_2D as _CONN_ROI_LOSS_2D,
)
from vis4d.config.util import get_callable_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import LossConnector, pred_key, remap_pred_keys
from vis4d.engine.loss_module import LossModule
from vis4d.model.track.qdtrack import FasterRCNNQDTrack
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.loss.common import smooth_l1_loss
from vis4d.op.track.qdtrack import QDTrackInstanceSimilarityLoss

from .faster_rcnn import (
    get_default_rcnn_box_codec_cfg,
    get_default_rpn_box_codec_cfg,
)

PRED_PREFIX = "detector_out"

CONN_BBOX_2D_TRAIN = {
    "images": K.images,
    "images_hw": K.input_hw,
    "frame_ids": K.frame_ids,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
    "boxes2d_track_ids": K.boxes2d_track_ids,
    "keyframes": "keyframes",
}

CONN_BBOX_2D_TEST = {
    "images": K.images,
    "images_hw": K.input_hw,
    "frame_ids": K.frame_ids,
}

CONN_RPN_LOSS_2D = {
    "cls_outs": pred_key(f"{PRED_PREFIX}.rpn.cls"),
    "reg_outs": pred_key(f"{PRED_PREFIX}.rpn.box"),
    "target_boxes": pred_key("key_target_boxes"),
    "images_hw": pred_key("key_images_hw"),
}

CONN_ROI_LOSS_2D = remap_pred_keys(_CONN_ROI_LOSS_2D, PRED_PREFIX)

CONN_TRACK_LOSS_2D = {
    "key_embeddings": pred_key("key_embeddings"),
    "ref_embeddings": pred_key("ref_embeddings"),
    "key_track_ids": pred_key("key_track_ids"),
    "ref_track_ids": pred_key("ref_track_ids"),
}


def get_qdtrack_cfg(
    num_classes: int | FieldReference,
    basemodel: ConfigDict,
    weights: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Get QD-Track model config."""
    ######################################################
    ##                        MODEL                     ##
    ######################################################
    anchor_generator = class_config(
        AnchorGenerator,
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
    )

    rpn_box_encoder, _ = get_default_rpn_box_codec_cfg()
    rcnn_box_encoder, _ = get_default_rcnn_box_codec_cfg()

    faster_rcnn_head = class_config(
        FasterRCNNHead,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
    )

    model = class_config(
        FasterRCNNQDTrack,
        num_classes=num_classes,
        basemodel=basemodel,
        faster_rcnn_head=faster_rcnn_head,
        weights=weights,
    )

    rpn_loss = class_config(
        RPNLoss,
        anchor_generator=anchor_generator,
        box_encoder=rpn_box_encoder,
        loss_bbox=get_callable_cfg(smooth_l1_loss, beta=1.0 / 9.0),
    )
    rcnn_loss = class_config(
        RCNNLoss,
        box_encoder=rcnn_box_encoder,
        num_classes=num_classes,
        loss_bbox=get_callable_cfg(smooth_l1_loss),
    )

    track_loss = class_config(QDTrackInstanceSimilarityLoss)

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
                "loss": track_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_TRACK_LOSS_2D
                ),
            },
        ],
    )

    return model, loss
