"""Detection utils."""
import os
from typing import List, Optional, Tuple

import torch
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.structures import Boxes, Instances

from openmt.struct import Boxes2D

from .base import BaseDetectorConfig

model_mapping = {
    "faster-rcnn": "COCO-Detection/faster_rcnn_",
    "mask-rcnn": "COCO-InstanceSegmentation/mask_rcnn_",
}

backbone_mapping = {
    "r101-fpn": "R_101_FPN_3x.yaml",
    "r101-c4": "R_101_C4_3x.yaml",
    "r101-dc5": "R_101_DC5_3x.yaml",
    "r50-fpn": "R_50_FPN_3x.yaml",
    "r50-c4": "R_50_C4_3x.yaml",
    "r50-dc5": "R_50_DC5_3x.yaml",
}


class D2GeneralizedRCNNConfig(BaseDetectorConfig):
    """Config for detectron2 rcnn-based models."""

    model_base: str
    override_mapping: Optional[bool] = False
    weights: Optional[str] = None
    num_classes: Optional[int]


def detections_to_box2d(detections: List[Instances]) -> List[Boxes2D]:
    """Convert d2 Instances representing detections to Boxes2D."""
    result = []
    for detection in detections:
        boxes, scores, cls = (
            detection.pred_boxes.tensor,
            detection.scores,
            detection.pred_classes,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, scores.unsqueeze(-1)], -1),
                class_ids=cls,
                image_wh=detection.image_size,
            )
        )
    return result


def proposal_to_box2d(proposals: List[Instances]) -> List[Boxes2D]:
    """Convert d2 Instances representing proposals to Boxes2D."""
    result = []
    for proposal in proposals:
        boxes, logits = (
            proposal.proposal_boxes.tensor,
            proposal.objectness_logits,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, logits.unsqueeze(-1)], -1),
                image_wh=proposal.image_size,
            )
        )
    return result


def target_to_instance(
    targets: List[Boxes2D], img_hw: Tuple[int, int]
) -> List[Instances]:
    """Convert Boxes2D representing targets to d2 Instances."""
    result = []
    for target in targets:
        boxes, cls, track_ids = (
            target.boxes[:, :4],
            target.class_ids,
            target.track_ids,
        )
        fields = dict(gt_boxes=Boxes(boxes), gt_classes=cls)
        if track_ids is not None:
            fields["track_ids"] = track_ids
        result.append(Instances(img_hw, **fields))
    return result


def model_to_detectron2(config: D2GeneralizedRCNNConfig) -> CfgNode:
    """Convert a Detector config to a detectron2 readable config."""
    cfg = get_cfg()

    # load detect base config, checkpoint
    detectron2_model_string = None
    if os.path.exists(config.model_base):
        base_cfg = config.model_base
    else:
        if config.override_mapping:
            detectron2_model_string = config.model_base
        else:
            model, backbone = config.model_base.split("/")
            detectron2_model_string = (
                model_mapping[model] + backbone_mapping[backbone]
            )
        base_cfg = model_zoo.get_config_file(detectron2_model_string)

    cfg.merge_from_file(base_cfg)

    # load checkpoint
    if config.weights is not None:
        if os.path.exists(config.weights):
            cfg.MODEL.WEIGHTS = config.weights
        elif (
            config.weights == "detectron2"
            and detectron2_model_string is not None
        ):
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                detectron2_model_string
            )
        else:
            raise ValueError(
                f"model weights path {config.weights} "
                f"not "
                f"found. If you're loading a detectron2 config from local, "
                f"please also specify a local checkpoint file"
            )

    # convert detect attributes
    if config.num_classes:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = config.num_classes

    return cfg
