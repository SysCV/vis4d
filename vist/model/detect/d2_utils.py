"""Detection utils."""
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.structures import Boxes, ImageList, Instances

from vist.struct import Boxes2D, Images

from ..base import BaseModelConfig

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


class D2TwoStageDetectorConfig(BaseModelConfig):
    """Config for detectron2 two stage models."""

    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]
    override_mapping: Optional[bool] = False
    num_classes: Optional[int]
    set_batchnorm_eval: bool = False
    weights: Optional[str]


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
            )
        )
    return result


def box2d_to_proposal(
    proposals: List[Boxes2D], imgs_wh: List[Tuple[int, int]]
) -> List[Instances]:
    """Convert Boxes2D representing proposals to d2 Instances."""
    result = []
    for proposal, img_wh in zip(proposals, imgs_wh):
        boxes, logits = (
            proposal.boxes[:, :4],
            proposal.boxes[:, -1],
        )
        fields = dict(proposal_boxes=Boxes(boxes), objectness_logits=logits)
        result.append(Instances((img_wh[1], img_wh[0]), **fields))
    return result


def target_to_instance(
    targets: List[Boxes2D], imgs_wh: List[Tuple[int, int]]
) -> List[Instances]:
    """Convert Boxes2D representing targets to d2 Instances."""
    result = []
    for target, img_wh in zip(targets, imgs_wh):
        boxes, cls, track_ids = (
            target.boxes,
            target.class_ids,
            target.track_ids,
        )
        fields = dict(gt_boxes=Boxes(boxes), gt_classes=cls)
        if track_ids is not None:
            fields["track_ids"] = track_ids
        result.append(Instances((img_wh[1], img_wh[0]), **fields))
    return result


def images_to_imagelist(images: Images) -> ImageList:
    """Convert Images to ImageList (switch from wh to hw for image sizes)."""
    return ImageList(
        images.tensor,
        image_sizes=[(wh[1], wh[0]) for wh in images.image_sizes],
    )


def model_to_detectron2(config: D2TwoStageDetectorConfig) -> CfgNode:
    """Convert a Detector config to a detectron2 readable config."""
    cfg = get_cfg()

    # load detect base config, checkpoint
    d2_model_string = None
    if os.path.exists(config.model_base):
        base_cfg = config.model_base
    else:
        if config.override_mapping:
            d2_model_string = config.model_base
        else:
            model, backbone = config.model_base.split("/")
            d2_model_string = model_mapping[model] + backbone_mapping[backbone]
        base_cfg = model_zoo.get_config_file(d2_model_string)

    cfg.merge_from_file(base_cfg)

    # prepare checkpoint path
    if config.weights is not None:
        if config.weights == "detectron2" and d2_model_string is not None:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(d2_model_string)
        else:
            cfg.MODEL.WEIGHTS = config.weights
    else:
        cfg.MODEL.WEIGHTS = ""

    # convert detect attributes
    if config.num_classes:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = config.num_classes

    # add keyword args in config
    if config.model_kwargs:
        for k, v in config.model_kwargs.items():
            attr = cfg
            partial_keys = k.split(".")
            partial_keys, last_key = partial_keys[:-1], partial_keys[-1]
            for part_k in partial_keys:
                attr = attr.get(part_k)
            attr_type = type(attr.get(last_key))
            attr.__setattr__(last_key, attr_type(v))
    return cfg
