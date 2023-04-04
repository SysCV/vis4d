"""Faseter R-CNN base model config."""
from __future__ import annotations

from torch import nn

from vis4d.config.util import ConfigDict, class_config
from ml_collections import FieldReference

from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key
from vis4d.engine.loss import WeightedMultiLoss

from vis4d.op.box.matchers import MaxIoUMatcher
from vis4d.op.box.samplers import RandomSampler
from vis4d.op.detect.rcnn import RCNNHead
from vis4d.op.detect.anchor_generator import AnchorGenerator
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss

from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder, DeltaXYWHBBoxDecoder

from vis4d.model.detect.faster_rcnn import FasterRCNN

# Data connectors
CONN_BBOX_2D_TRAIN = {
    K.images: K.images,
    K.input_hw: K.input_hw,
    K.boxes2d: K.boxes2d,
    K.boxes2d_classes: K.boxes2d_classes,
}

CONN_BBOX_2D_TEST = {
    **CONN_BBOX_2D_TRAIN,
    "original_hw": "original_hw",
}

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
    "pred_sampled_proposals": pred_key("sampled_proposals"),
}


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
    )


def get_default_rpn_box_codec(
    target_means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    target_stds: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[nn.Module, nn.Module]:
    """Get the default bounding box encoder and decoder for RPN."""
    return (
        DeltaXYWHBBoxEncoder(target_means, target_stds),
        DeltaXYWHBBoxDecoder(target_means, target_stds),
    )


def get_default_rcnn_box_codec(
    target_means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    target_stds: tuple[float, ...] = (0.1, 0.1, 0.2, 0.2),
) -> tuple[nn.Module, nn.Module]:
    """Get the default bounding box encoder and decoder for RCNN."""
    return (
        DeltaXYWHBBoxEncoder(target_means, target_stds),
        DeltaXYWHBBoxDecoder(target_means, target_stds),
    )


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default bounding box matcher."""
    return MaxIoUMatcher(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    )


def get_default_box_sampler() -> RandomSampler:
    """Get default bounding box sampler."""
    return RandomSampler(batch_size=512, positive_fraction=0.25)


def get_default_roi_head(num_classes: int) -> RCNNHead:
    """Get default ROI head."""
    return RCNNHead(num_classes=num_classes)


def get_default_faster_rcnn_loss(
    anchor_generator: AnchorGenerator,
    rpn_box_encoder: nn.Module,
    rcnn_box_encoder: nn.Module,
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


def get_model_cfg(
    num_classes: FieldReference | int,
    backbone: ConfigDict,
) -> tuple[ConfigDict, ConfigDict]:
    """Return default config for faster_rcnn model and loss."""
    ######################################################
    ##                        MODEL                     ##
    ######################################################
    anchor_generator = get_default_anchor_generator()
    rpn_box_encoder, rpn_box_decoder = get_default_rpn_box_codec()
    rcnn_box_encoder, rcnn_box_decoder = get_default_rcnn_box_codec()

    box_matcher = get_default_box_matcher()
    box_sampler = get_default_box_sampler()

    roi_head = class_config(get_default_roi_head, num_classes=num_classes)

    model = class_config(
        FasterRCNN,
        backbone=backbone,
        anchor_generator=anchor_generator,
        rpn_box_decoder=rpn_box_decoder,
        rcnn_box_decoder=rcnn_box_decoder,
        box_matcher=box_matcher,
        box_sampler=box_sampler,
        roi_head=roi_head,
        # weights="mmdet",
    )

    ######################################################
    ##                      LOSS                        ##
    ######################################################
    loss = class_config(
        get_default_faster_rcnn_loss,
        rpn_box_encoder=rpn_box_encoder,
        rcnn_box_encoder=rcnn_box_encoder,
        anchor_generator=anchor_generator,
    )
    return model, loss
