"""Utilities for mmdet wrapper."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from mmcv import Config as MMConfig

from vist.struct import Boxes2D, Images, LossesType

from ..base import BaseModelConfig

MMDetMetaData = Dict[str, Union[Tuple[int, int, int], bool, float]]


class MMTwoStageDetectorConfig(BaseModelConfig):
    """Config for mmdetection two stage models."""

    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]
    num_classes: Optional[int]
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    backbone_output_names: Optional[List[str]]
    weights: Optional[str]


def get_img_metas(images: Images) -> List[MMDetMetaData]:
    """Create image metadata in mmdetection format."""
    img_metas = []
    _, c, padh, padw = images.tensor.shape  # type: Tuple[int, int, int, int]
    for i in range(len(images)):
        meta = dict()  # type: MMDetMetaData
        w, h = images.image_sizes[i]
        meta["img_shape"] = meta["ori_shape"] = (h, w, c)
        meta["scale_factor"] = 1.0
        meta["flip"] = False
        meta["pad_shape"] = (padh, padw, c)
        img_metas.append(meta)

    return img_metas


def proposals_from_mmdet(proposals: List[torch.Tensor]) -> List[Boxes2D]:
    """Convert mmdetection proposals to VisT format."""
    proposals_boxes2d = []
    for proposal in proposals:
        proposals_boxes2d.append(Boxes2D(proposal))
    return proposals_boxes2d


def proposals_to_mmdet(proposals: List[Boxes2D]) -> List[torch.Tensor]:
    """Convert VisT format proposals to mmdetection."""
    proposal_tensors = []
    for proposal in proposals:
        proposal_tensors.append(proposal.boxes)
    return proposal_tensors


def detections_from_mmdet(
    bboxes: List[torch.Tensor], labels: List[torch.Tensor]
) -> List[Boxes2D]:
    """Convert mmdetection detections to VisT format."""
    detections_boxes2d = []
    for bbox, label in zip(bboxes, labels):
        if not label.device == bbox.device:
            label = label.to(bbox.device)  # pragma: no cover
        detections_boxes2d.append(Boxes2D(bbox, label))
    return detections_boxes2d


def results_from_mmdet(
    detections: List[torch.Tensor], device: torch.device
) -> List[Boxes2D]:
    """Convert mmdetection bbox_results to VisT format."""
    detections_boxes2d = []
    for detection in detections:
        bboxes = torch.from_numpy(np.vstack(detection)).to(device)  # Nx5
        labels = [
            torch.full((bbox.shape[0],), i, dtype=torch.int32, device=device)
            for i, bbox in enumerate(detection)
        ]
        labels = torch.cat(labels)
        detections_boxes2d.append(Boxes2D(bboxes, labels))
    return detections_boxes2d


def targets_to_mmdet(
    targets: List[Boxes2D],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Convert VisT targets to mmdetection compatible format."""
    gt_bboxes = [t.boxes for t in targets]

    gt_labels = [t.class_ids for t in targets]
    return gt_bboxes, gt_labels


def load_config_from_mmdet(url: str) -> str:
    """Get config from mmdetection GitHub repository."""
    full_url = (
        "https://raw.githubusercontent.com/"
        "open-mmlab/mmdetection/master/configs/" + url
    )
    response = requests.get(full_url)
    assert (
        response.status_code == 200
    ), f"Request to {full_url} failed with code {response.status_code}!"
    return response.text


def get_mmdet_config(config: MMTwoStageDetectorConfig) -> MMConfig:
    """Convert a Detector config to a mmdet readable config."""
    if os.path.exists(config.model_base):
        cfg = MMConfig.fromfile(config.model_base)
        if cfg.get("model"):
            cfg = cfg["model"]
    elif config.model_base.startswith("mmdet://"):
        ext = os.path.splitext(config.model_base)[1]
        cfg = MMConfig.fromstring(
            load_config_from_mmdet(config.model_base.strip("mmdet://")), ext
        ).model
    else:
        raise FileNotFoundError(
            f"MMDetection config not found: {config.model_base}"
        )

    if config.num_classes:
        cfg["roi_head"]["bbox_head"]["num_classes"] = config.num_classes

    # add keyword args in config
    if config.model_kwargs:
        for k, v in config.model_kwargs.items():
            attr = cfg
            partial_keys = k.split(".")
            partial_keys, last_key = partial_keys[:-1], partial_keys[-1]
            for part_k in partial_keys:
                attr = attr.get(part_k)
            if attr.get(last_key) is not None:
                attr[last_key] = type(attr.get(last_key))(v)
            else:
                attr[last_key] = v

    return cfg


def _parse_losses(losses: Dict[str, torch.Tensor]) -> LossesType:
    """Parse losses to a scalar tensor."""
    log_vars = dict()
    for name, value in losses.items():
        if "loss" in name:
            if isinstance(value, torch.Tensor):
                log_vars[name] = value.mean()
            elif isinstance(value, list):
                log_vars[name] = sum(_loss.mean() for _loss in value)
            else:
                raise ValueError(f"{name} is not a tensor or list of tensors")

    return log_vars
