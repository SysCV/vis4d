"""Detection utils."""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from vis4d.common.utils.imports import DETECTRON2_AVAILABLE

from vis4d.common.mask import paste_masks_in_image
from vis4d.struct import (
    Boxes2D,
    DictStrAny,
    Images,
    InputSample,
    InstanceMasks,
)

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


def segmentations_to_bitmask(
    inputs: InputSample,
    segmentations: List[Instances],
    detections: List[Boxes2D],
) -> List[InstanceMasks]:
    """Convert d2 Instances representing segmentations to Masks."""
    result = []
    for inp, segmentation, det in zip(inputs, segmentations, detections):
        pred_mask = paste_masks_in_image(
            segmentation.pred_masks.squeeze(1),
            det.boxes,
            inp.images.image_sizes[0],
        )
        result.append(
            InstanceMasks(
                pred_mask,
                class_ids=segmentation.pred_classes,
                score=segmentation.scores,
                detections=det,
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
    gt_boxes: Sequence[Boxes2D],
    imgs_wh: List[Tuple[int, int]],
    gt_masks: Optional[Sequence[InstanceMasks]] = None,
) -> List[Instances]:
    """Convert Boxes2D and Masks representing targets to d2 Instances."""
    result = []
    if gt_masks is None:
        gt_masks = [None] * len(gt_boxes)  # type: ignore
    for gt_box, gt_mask, img_wh in zip(gt_boxes, gt_masks, imgs_wh):
        boxes, cls, track_ids = (
            gt_box.boxes,
            gt_box.class_ids,
            gt_box.track_ids,
        )
        fields = dict(gt_boxes=Boxes(boxes), gt_classes=cls)
        if track_ids is not None:  # pragma: no cover
            fields["track_ids"] = track_ids
        if gt_mask is not None and len(gt_mask) > 0:
            fields["gt_masks"] = BitMasks(gt_mask.masks)
        result.append(Instances((img_wh[1], img_wh[0]), **fields))
    return result


def images_to_imagelist(images: Images) -> ImageList:
    """Convert Images to ImageList (switch from wh to hw for image sizes)."""
    return ImageList(
        images.tensor,
        image_sizes=[(wh[1], wh[0]) for wh in images.image_sizes],
    )


def model_to_detectron2(
    model_base: str,
    model_kwargs: Optional[DictStrAny] = None,
    override_mapping: Optional[bool] = False,
    weights: Optional[str] = None,
    category_mapping: Optional[Dict[str, int]] = None,
) -> CfgNode:
    """Convert a Detector config to a detectron2 readable config."""
    cfg = get_cfg()

    # load detect base config, checkpoint
    d2_model_string = None
    if os.path.exists(model_base):
        base_cfg = model_base
    else:
        if override_mapping:
            d2_model_string = model_base
        else:
            model, backbone = model_base.split("/")
            d2_model_string = model_mapping[model] + backbone_mapping[backbone]
        base_cfg = model_zoo.get_config_file(d2_model_string)

    cfg.merge_from_file(base_cfg)

    # prepare checkpoint path
    if weights is not None:
        if weights == "detectron2" and d2_model_string is not None:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(d2_model_string)
        else:  # pragma: no cover
            cfg.MODEL.WEIGHTS = weights
    else:
        cfg.MODEL.WEIGHTS = ""

    # convert detect attributes
    assert category_mapping is not None
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category_mapping)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(category_mapping)

    # add keyword args in config
    if model_kwargs:
        for k, v in model_kwargs.items():
            attr = cfg
            partial_keys = k.split(".")
            partial_keys, last_key = partial_keys[:-1], partial_keys[-1]
            for part_k in partial_keys:
                attr = attr.get(part_k)
            attr_type = type(attr.get(last_key))
            attr.__setattr__(last_key, attr_type(v))
    return cfg
