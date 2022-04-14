"""Utilities for models, mm wrappers."""
import os
import re
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import requests
import torch

from vis4d.struct import (
    Boxes2D,
    DictStrAny,
    Images,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    ModelOutput,
    NDArrayF64,
    NDArrayUI8,
    SemanticMasks,
    TLabelInstance,
)

from .base import BaseModel

try:
    from mmcv import Config as MMConfig
    from mmcv.runner.checkpoint import load_checkpoint

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmdet.core.mask import BitmapMasks

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False
    BitmapMasks = None


BDD100K_MODEL_PREFIX = "https://dl.cv.ethz.ch/bdd100k/"
MM_MODEL_MAP = {
    "mmdet://": "https://download.openmmlab.com/mmdetection/v2.0/",
    "mmseg://": "https://download.openmmlab.com/mmsegmentation/v0.5/",
}
MM_CFG_MAP = {
    "mmdet://": "syscv/mmdetection/master/configs/",
    "mmseg://": "open-mmlab/mmsegmentation/master/configs/",
}
MM_ZIP_MAP = {
    "mmdet://": "mmdetection-master/configs/",
    "mmseg://": "mmsegmentation-master/configs/",
}

MMDetMetaData = Dict[str, Union[Tuple[int, int, int], bool, NDArrayF64]]
MMDetResult = List[torch.Tensor]
MMSegmResult = List[List[NDArrayUI8]]
MMDetResults = Union[List[MMDetResult], List[Tuple[MMDetResult, MMSegmResult]]]
MMSegResults = Union[List[NDArrayUI8], torch.Tensor]


def get_img_metas(images: Images) -> List[MMDetMetaData]:
    """Create image metadata in mmdetection format."""
    img_metas = []
    _, c, padh, padw = images.tensor.shape  # type: Tuple[int, int, int, int]
    for i in range(len(images)):
        meta: MMDetMetaData = {}
        w, h = images.image_sizes[i]
        meta["img_shape"] = meta["ori_shape"] = (h, w, c)
        meta["scale_factor"] = np.ones(4, dtype=np.float64)
        meta["flip"] = False
        meta["pad_shape"] = (padh, padw, c)
        img_metas.append(meta)

    return img_metas


def proposals_from_mmdet(
    proposals: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
) -> List[Boxes2D]:
    """Convert mmdetection proposals to Vis4D format."""
    proposals_boxes2d = []
    for proposal in proposals:
        if isinstance(proposal, tuple):
            proposals_boxes2d.append(Boxes2D(*proposal))
        else:
            proposals_boxes2d.append(Boxes2D(proposal))
    return proposals_boxes2d


def proposals_to_mmdet(proposals: List[Boxes2D]) -> List[torch.Tensor]:
    """Convert Vis4D format proposals to mmdetection."""
    proposal_tensors = []
    for proposal in proposals:
        proposal_tensors.append(proposal.boxes)
    return proposal_tensors


def detections_from_mmdet(
    bboxes: List[torch.Tensor], labels: List[torch.Tensor]
) -> List[Boxes2D]:
    """Convert mmdetection detections to Vis4D format."""
    detections_boxes2d = []
    for bbox, label in zip(bboxes, labels):
        if not label.device == bbox.device:
            label = label.to(bbox.device)  # pragma: no cover
        detections_boxes2d.append(Boxes2D(bbox, label))
    return detections_boxes2d


def segmentations_from_mmdet(
    masks: List[MMSegmResult], boxes: List[Boxes2D], device: torch.device
) -> List[InstanceMasks]:
    """Convert mmdetection segmentations to Vis4D format."""
    segmentations_masks = []
    for mask_res, box_res in zip(masks, boxes):
        mask = segmentation_from_mmdet_results(mask_res, box_res, device)
        segmentations_masks.append(mask)
    return segmentations_masks


def segmentation_from_mmdet_results(
    segmentation: MMSegmResult, boxes: Boxes2D, device: torch.device
) -> InstanceMasks:
    """Convert segm_result to Vis4D format."""
    segms: List[NDArrayUI8] = [
        np.stack(segm) if len(segm) != 0 else np.empty_like(segm)
        for segm in segmentation
    ]
    if len(segms) == 0 or sum([len(segm) for segm in segmentation]) == 0:
        return InstanceMasks.empty(device)  # pragma: no cover
    masks_list, labels_list = [], []  # type: ignore
    for class_id in boxes.class_ids:
        masks_list.append(
            torch.from_numpy(segms[class_id][labels_list.count(class_id)])
            .type(torch.uint8)
            .to(device)
        )
        labels_list.append(class_id)
    masks = torch.stack(masks_list)
    labels = torch.stack(labels_list)
    scores = boxes.score
    return InstanceMasks(masks, labels, score=scores, detections=boxes)


def masks_to_mmdet_masks(masks: Sequence[InstanceMasks]) -> BitmapMasks:
    """Convert Vis4D Masks to mmdetection BitmapMasks."""
    return [BitmapMasks(m.to_ndarray(), m.height, m.width) for m in masks]


def targets_to_mmdet(
    targets: LabelInstances,
) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], Optional[Sequence[BitmapMasks]]
]:
    """Convert Vis4D targets to mmdetection compatible format."""
    gt_bboxes = [t.boxes for t in targets.boxes2d]
    gt_labels = [t.class_ids for t in targets.boxes2d]
    if all(len(t) == 0 for t in targets.instance_masks):  # pragma: no cover
        gt_masks = None
    else:
        gt_masks = masks_to_mmdet_masks(targets.instance_masks)
    return gt_bboxes, gt_labels, gt_masks


def results_from_mmseg(
    results: MMSegResults, device: torch.device
) -> List[SemanticMasks]:
    """Convert mmsegmentation seg_pred to Vis4D format."""
    masks = []
    for result in results:
        if isinstance(result, torch.Tensor):
            mask = result.unsqueeze(0).byte()
        else:  # pragma: no cover
            mask = torch.tensor([result], device=device).byte()
        masks.append(SemanticMasks(mask).to_nhw_mask())
    return masks


def targets_to_mmseg(images: Images, targets: LabelInstances) -> torch.Tensor:
    """Convert Vis4D targets to mmsegmentation compatible format."""
    if len(targets.semantic_masks) > 1:
        # pad masks to same size for batching
        targets.semantic_masks = SemanticMasks.pad(
            targets.semantic_masks, images.tensor.shape[-2:][::-1]
        )
    return torch.stack(
        [t.to_hwc_mask() for t in targets.semantic_masks]
    ).unsqueeze(1)


def load_model_checkpoint(
    model: BaseModel,
    weights: str,
    rev_keys: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Load MM model checkpoint."""
    if rev_keys is None:  # pragma: no cover
        rev_keys = [(r"^module\.", "")]
    if re.compile(r"^mm(det|seg)://").search(weights):
        pre = weights[:8]
        weights = MM_MODEL_MAP[pre] + weights.split(pre)[-1]
        load_checkpoint(model, weights, revise_keys=rev_keys)
    elif weights.startswith("bdd100k://"):
        weights = BDD100K_MODEL_PREFIX + weights.split("bdd100k://")[-1]
        load_checkpoint(model, weights, revise_keys=rev_keys)
    else:  # pragma: no cover
        load_checkpoint(model, weights)


def load_config_from_mm(url: str, mm_base: str) -> str:
    """Get config from mmdetection GitHub repository."""
    full_url = "https://raw.githubusercontent.com/" + mm_base + url
    response = requests.get(full_url)
    assert (
        response.status_code == 200
    ), f"Request to {full_url} failed with code {response.status_code}!"
    return response.text


def load_config(path: str, key: str = "model") -> MMConfig:
    """Load config either from file or from URL."""
    if os.path.exists(path):
        cfg = MMConfig.fromfile(path)
    elif re.compile(r"^mm(det|seg)://").search(path):
        pre = path[:8]
        cfg_content = load_config_from_mm(path.split(pre)[-1], MM_CFG_MAP[pre])
        if cfg_content.find("_base_") >= 0:
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as temp_config_dir:
                # download configs
                url_tmp = MM_CFG_MAP[pre].replace("master/configs/", "archive")
                with ZipFile(
                    BytesIO(
                        urlopen(
                            f"https://github.com/{url_tmp}/master.zip"
                        ).read()
                    )
                ) as zipfile:
                    zipfile.extractall(path=temp_config_dir)
                os.chdir(os.path.join(temp_config_dir, MM_ZIP_MAP[pre]))
                cfg = MMConfig.fromfile(path.replace(pre, ""))
                os.chdir(cwd)
        else:
            cfg = MMConfig.fromstring(cfg_content, os.path.splitext(path)[1])
    else:
        raise FileNotFoundError(f"MM config not found: {path}")
    assert key in cfg
    return cfg[key]


def _parse_losses(
    losses: Dict[str, torch.Tensor], prefix: Optional[str] = None
) -> LossesType:
    """Parse losses to a scalar tensor."""
    log_vars = {}
    for name, value in losses.items():
        if "loss" in name:
            if prefix is not None:  # pragma: no cover
                name = f"{prefix}.{name}"
            if isinstance(value, torch.Tensor):
                log_vars[name] = value.mean()
            elif isinstance(value, list):
                log_vars[name] = sum(_loss.mean() for _loss in value)
            else:
                raise ValueError(f"{name} is not a tensor or list of tensors")

    return log_vars


def set_attr(  # type: ignore
    attr: Any, partial_keys: List[str], last_key: str, value: Any
) -> None:
    """Set specific attribute in config."""
    for i, part_k in enumerate(partial_keys):
        if isinstance(attr, list):  # pragma: no cover
            for attr_item in attr:
                set_attr(attr_item, partial_keys[i:], last_key, value)
            return
        attr = attr.get(part_k)
    attr[last_key] = value


def add_keyword_args(model_kwargs: DictStrAny, cfg: MMConfig) -> None:
    """Add keyword args in config."""
    for k, v in model_kwargs.items():
        partial_keys = k.split(".")
        partial_keys, last_key = partial_keys[:-1], partial_keys[-1]
        set_attr(cfg, partial_keys, last_key, v)


def postprocess_predictions(
    inputs: InputSample,
    predictions: Dict[str, List[TLabelInstance]],
    clip_to_image: bool = True,
    resolve_overlap: bool = True,
) -> None:
    """Postprocess predictions."""
    for values in predictions.values():
        for inp, v in zip(inputs, values):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            v.postprocess(
                input_size,
                inp.images.image_sizes[0],
                clip_to_image,
                resolve_overlap,
            )


def predictions_to_scalabel(
    predictions: Dict[str, List[TLabelInstance]],
    idx_to_class: Optional[Dict[int, str]] = None,
) -> ModelOutput:
    """Convert predictions into ModelOutput (Scalabel)."""
    outputs = {}
    for key, values in predictions.items():
        outputs[key] = [
            v.to(torch.device("cpu")).to_scalabel(idx_to_class) for v in values
        ]
    return outputs
