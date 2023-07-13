"""CC-3DT model config."""
from __future__ import annotations

from ml_collections import ConfigDict, FieldReference

from vis4d.config import class_config
from vis4d.config.util import get_callable_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import (
    LossConnector,
    data_key,
    pred_key,
    remap_pred_keys,
)
from vis4d.engine.loss_module import LossModule
from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.detect3d.qd_3dt import Box3DUncertaintyLoss
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RCNNHead, RCNNLoss
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.loss.common import smooth_l1_loss
from vis4d.op.track.qdtrack import QDTrackInstanceSimilarityLoss
from vis4d.state.track3d.cc_3dt import CC3DTrackGraph

from .faster_rcnn import (
    get_default_rcnn_box_codec_cfg,
    get_default_rpn_box_codec_cfg,
)
from .qdtrack import CONN_ROI_LOSS_2D as _CONN_ROI_LOSS_2D
from .qdtrack import CONN_TRACK_LOSS_2D as _CONN_TRACK_LOSS_2D

PRED_PREFIX = "qdtrack_out"

CONN_RPN_LOSS_2D = {
    "cls_outs": pred_key(f"{PRED_PREFIX}.detector_out.rpn.cls"),
    "reg_outs": pred_key(f"{PRED_PREFIX}.detector_out.rpn.box"),
    "target_boxes": pred_key(f"{PRED_PREFIX}.key_target_boxes"),
    "images_hw": pred_key(f"{PRED_PREFIX}.key_images_hw"),
}

CONN_ROI_LOSS_2D = remap_pred_keys(_CONN_ROI_LOSS_2D, PRED_PREFIX)

CONN_TRACK_LOSS_2D = remap_pred_keys(_CONN_TRACK_LOSS_2D, PRED_PREFIX)

CONN_DET_3D_LOSS = {
    "pred": pred_key("detector_3d_out"),
    "target": pred_key("detector_3d_target"),
    "labels": pred_key("detector_3d_labels"),
}

CONN_BBOX_3D_TRAIN = {
    "images": K.images,
    "images_hw": K.input_hw,
    "intrinsics": K.intrinsics,
    "boxes2d": K.boxes2d,
    "boxes3d": K.boxes3d,
    "boxes3d_classes": K.boxes3d_classes,
    "boxes3d_track_ids": K.boxes3d_track_ids,
    "keyframes": "keyframes",
}

CONN_BBOX_3D_TEST = {
    "images": K.images,
    "images_hw": K.original_hw,
    "intrinsics": K.intrinsics,
    "extrinsics": K.extrinsics,
    "frame_ids": K.frame_ids,
}

CONN_NUSC_DET3D_EVAL = {
    "tokens": data_key("token"),
    "boxes_3d": pred_key("boxes_3d"),
    "velocities": pred_key("velocities"),
    "class_ids": pred_key("class_ids"),
    "scores_3d": pred_key("scores_3d"),
}

CONN_NUSC_TRACK3D_EVAL = {
    "tokens": data_key("token"),
    "boxes_3d": pred_key("boxes_3d"),
    "velocities": pred_key("velocities"),
    "class_ids": pred_key("class_ids"),
    "scores_3d": pred_key("scores_3d"),
    "track_ids": pred_key("track_ids"),
}


def get_cc_3dt_cfg(
    num_classes: int | FieldReference,
    basemodel: ConfigDict,
    detection_range: list[float] | FieldReference | None = None,
    fps: int | FieldReference = 2,
    weights: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Get CC-3DT model config.

    Args:
        num_classes (FieldReference | int): Number of classes.
        basemodel (ConfigDict): Base model config.
        detection_range (list[float]| FieldReference | None, optional):
            Detection range. Defaults to None.
        fps (int | FieldReference, optional): FPS. Defaults to 2.
        weights (str | None, optional): Weights to load. Defaults to None.
    """
    ######################################################
    ##                        MODEL                     ##
    ######################################################
    anchor_generator = class_config(
        AnchorGenerator,
        scales=[4, 8],
        ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
        strides=[4, 8, 16, 32, 64],
    )

    roi_head = class_config(
        RCNNHead,
        num_shared_convs=4,
        num_classes=num_classes,
    )

    faster_rcnn_head = class_config(
        FasterRCNNHead,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        roi_head=roi_head,
    )

    track_graph = class_config(
        CC3DTrackGraph,
        motion_model="KF3D",
        detection_range=detection_range,
        fps=fps,
    )

    model = class_config(
        FasterRCNNCC3DT,
        num_classes=num_classes,
        basemodel=basemodel,
        faster_rcnn_head=faster_rcnn_head,
        track_graph=track_graph,
        weights=weights,
    )

    ######################################################
    ##                      LOSS                        ##
    ######################################################
    rpn_box_encoder, _ = get_default_rpn_box_codec_cfg()
    rcnn_box_encoder, _ = get_default_rcnn_box_codec_cfg()

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
        loss_bbox=get_callable_cfg(smooth_l1_loss, beta=1.0 / 9.0),
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
                "weight": 5.0,
            },
            {
                "loss": track_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_TRACK_LOSS_2D
                ),
            },
            {
                "loss": class_config(Box3DUncertaintyLoss),
                "connector": class_config(
                    LossConnector, key_mapping=CONN_DET_3D_LOSS
                ),
            },
        ],
    )

    return model, loss
