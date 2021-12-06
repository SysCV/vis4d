"""Utilities for mmseg wrapper."""

import os
from typing import Dict, List, Optional, Tuple, Union

import requests
import torch
import torch.nn.functional as F

try:
    from mmcv import Config as MMConfig

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False


from vis4d.common.mmdet_utils import MMDetMetaData, add_keyword_args
from vis4d.model.base import BaseModelConfig
from vis4d.struct import LabelInstances, NDArrayUI8, SemanticMasks

MMResults = Union[List[NDArrayUI8], torch.Tensor]


class MMEncDecSegmentorConfig(BaseModelConfig):
    """Config for mmsegmentation encoder-decoder models."""

    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    backbone_output_names: Optional[List[str]]
    weights: Optional[str]


def segmentations_from_mmseg(
    masks: torch.Tensor, device: torch.device
) -> List[SemanticMasks]:  # pragma: no cover
    """Convert mmsegmentation segmentations to Vis4D format."""
    return [
        SemanticMasks(mask).to_nhw_mask().to(device)
        for mask in F.softmax(masks, dim=1)
    ]


def results_from_mmseg(
    results: MMResults, img_metas: List[MMDetMetaData], device: torch.device
) -> List[SemanticMasks]:
    """Convert mmsegmentation seg_pred to Vis4D format."""
    masks = []
    for result, img_meta in zip(results, img_metas):
        ori_h, ori_w = img_meta["ori_shape"][:2]  # type: ignore
        result = result[:ori_h, :ori_w]
        if isinstance(result, torch.Tensor):
            mask = result.unsqueeze(0).byte()
        else:
            mask = torch.tensor([result], device=device).byte()
        masks.append(SemanticMasks(mask).to_nhw_mask())
    return masks


def targets_to_mmseg(targets: LabelInstances) -> torch.Tensor:
    """Convert Vis4D targets to mmsegmentation compatible format."""
    return torch.stack(
        [t.to_hwc_mask() for t in targets.semantic_masks]
    ).unsqueeze(1)


def load_config_from_mmseg(url: str) -> str:
    """Get config from mmsegmentation GitHub repository."""
    full_url = (
        "https://raw.githubusercontent.com/"
        "open-mmlab/mmsegmentation/master/configs/" + url
    )
    response = requests.get(full_url)
    assert (
        response.status_code == 200
    ), f"Request to {full_url} failed with code {response.status_code}!"
    return response.text


def get_mmseg_config(config: MMEncDecSegmentorConfig) -> MMConfig:
    """Convert a Segmentor config to a mmseg readable config."""
    if os.path.exists(config.model_base):
        cfg = MMConfig.fromfile(config.model_base)
        if cfg.get("model"):
            cfg = cfg["model"]
    elif config.model_base.startswith("mmseg://"):
        ex = os.path.splitext(config.model_base)[1]
        cfg = MMConfig.fromstring(
            load_config_from_mmseg(config.model_base.split("mmseg://")[-1]), ex
        ).model
    else:
        raise FileNotFoundError(
            f"MMSegmentation config not found: {config.model_base}"
        )

    # convert segmentor attributes
    assert config.category_mapping is not None
    if isinstance(cfg["decode_head"], list):
        if isinstance(cfg["decode_head"], list):
            for dec_head in cfg["decode_head"]:
                dec_head["num_classes"] = len(config.category_mapping)
        else:
            cfg["decode_head"]["num_classes"] = len(config.category_mapping)
    if "auxiliary_head" in cfg:
        if isinstance(cfg["auxiliary_head"], list):
            for aux_head in cfg["auxiliary_head"]:
                aux_head["num_classes"] = len(config.category_mapping)
        else:
            cfg["auxiliary_head"]["num_classes"] = len(config.category_mapping)

    if config.model_kwargs:
        add_keyword_args(config, cfg)
    return cfg
