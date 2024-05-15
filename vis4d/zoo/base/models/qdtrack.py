"""QD-Track model config."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference

from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import LossConnector, pred_key, remap_pred_keys
from vis4d.engine.loss_module import LossModule
from vis4d.model.adapter import ModelExpEMAAdapter
from vis4d.model.track.qdtrack import FasterRCNNQDTrack, YOLOXQDTrack
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.poolers import MultiScaleRoIAlign
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.detect.yolox import YOLOXHeadLoss
from vis4d.op.loss.common import smooth_l1_loss
from vis4d.op.track.qdtrack import (
    QDSimilarityHead,
    QDTrackHead,
    QDTrackInstanceSimilarityLoss,
)
from vis4d.zoo.base import get_callable_cfg
from vis4d.zoo.base.models.faster_rcnn import (
    CONN_ROI_LOSS_2D as _CONN_ROI_LOSS_2D,
)
from vis4d.zoo.base.models.faster_rcnn import (
    get_default_rcnn_box_codec_cfg,
    get_default_rpn_box_codec_cfg,
)

from .yolox import get_yolox_model_cfg

PRED_PREFIX = "detector_out"

CONN_BBOX_2D_TRAIN = {
    "images": K.images,
    "images_hw": K.input_hw,
    "original_hw": K.original_hw,
    "frame_ids": K.frame_ids,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
    "boxes2d_track_ids": K.boxes2d_track_ids,
    "keyframes": "keyframes",
}

CONN_BBOX_2D_TEST = {
    "images": K.images,
    "images_hw": K.input_hw,
    "original_hw": K.original_hw,
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

CONN_YOLOX_LOSS_2D = {
    "cls_outs": pred_key(f"{PRED_PREFIX}.cls_score"),
    "reg_outs": pred_key(f"{PRED_PREFIX}.bbox_pred"),
    "obj_outs": pred_key(f"{PRED_PREFIX}.objectness"),
    "target_boxes": pred_key("key_target_boxes"),
    "target_class_ids": pred_key("key_target_classes"),
    "images_hw": pred_key("key_images_hw"),
}


def get_qdtrack_cfg(
    num_classes: int | FieldReference,
    basemodel: ConfigDict,
    weights: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Get QDTrack model config."""
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


def get_qdtrack_yolox_cfg(
    num_classes: int | FieldReference,
    model_type: str,
    use_ema: bool = True,
    weights: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Get QDTrack YOLOX model config."""
    ######################################################
    ##                        MODEL                     ##
    ######################################################
    basemodel, fpn, yolox_head = get_yolox_model_cfg(num_classes, model_type)
    if model_type == "tiny":
        in_dim = 96
    elif model_type == "small":
        in_dim = 128
    elif model_type == "large":
        in_dim = 256
    elif model_type == "xlarge":
        in_dim = 320
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    model = class_config(
        YOLOXQDTrack,
        num_classes=num_classes,
        basemodel=basemodel,
        fpn=fpn,
        yolox_head=yolox_head,
        qdtrack_head=class_config(
            QDTrackHead,
            similarity_head=class_config(
                QDSimilarityHead,
                proposal_pooler=MultiScaleRoIAlign(
                    resolution=(7, 7), strides=[8, 16, 32], sampling_ratio=0
                ),
                in_dim=in_dim,
            ),
        ),
        weights=weights,
    )
    if use_ema:
        model = class_config(ModelExpEMAAdapter, model=model)

    track_loss = class_config(QDTrackInstanceSimilarityLoss)

    loss = class_config(
        LossModule,
        losses=[
            {
                "loss": class_config(YOLOXHeadLoss, num_classes=num_classes),
                "connector": class_config(
                    LossConnector, key_mapping=CONN_YOLOX_LOSS_2D
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
