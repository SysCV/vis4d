"""Utilities for mmseg wrapper."""

import os
from typing import Dict, List, Optional, Tuple, Union

import requests
import torch
import torch.nn.functional as F

try:
    from mmcv import Config as MMConfig

    MMCV_INSTALLED = True
except:
    MMCV_INSTALLED = False  # pragma: no cover


from vist.struct import InputSample, NDArrayUI8, SemanticMasks

from ..base import BaseModelConfig

MMResults = List[NDArrayUI8]


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
    """Convert mmsegmentation segmentations to VisT format."""
    return [
        SemanticMasks(mask).to_nhw_mask().to(device)
        for mask in F.softmax(masks, dim=1)
    ]


def results_from_mmseg(
    results: MMResults, device: torch.device
) -> List[SemanticMasks]:
    """Convert mmsegmentation seg_pred to VisT format."""
    return [
        SemanticMasks(
            torch.tensor([result], device=device).byte()
        ).to_nhw_mask()
        for result in results
    ]


def targets_to_mmseg(targets: InputSample) -> torch.Tensor:
    """Convert VisT targets to mmsegmentation compatible format."""
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
    if os.path.exists(config.model_base):  # pragma: no cover
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
    cfg["decode_head"]["num_classes"] = len(config.category_mapping)
    if "auxiliary_head" in cfg:
        if isinstance(cfg["auxiliary_head"], list):  # pragma: no cover
            for aux_head in cfg["auxiliary_head"]:
                aux_head["num_classes"] = len(config.category_mapping)
        else:
            cfg["auxiliary_head"]["num_classes"] = len(config.category_mapping)

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
                attr[last_key] = v  # pragma: no cover

    return cfg
