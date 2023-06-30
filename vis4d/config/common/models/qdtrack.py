"""QD-Track model config."""
from __future__ import annotations

from vis4d.config.common.models.faster_rcnn import (
    CONN_ROI_LOSS_2D as _CONN_ROI_LOSS_2D,
)
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import pred_key, remap_pred_keys

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
