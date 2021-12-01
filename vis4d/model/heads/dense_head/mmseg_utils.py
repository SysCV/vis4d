"""Utilities for mmseg wrapper."""

import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

try:
    from mmcv import Config as MMConfig

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False


from vis4d.model.detect.mmdet_utils import MMDetMetaData, add_keyword_args
from vis4d.model.segment.mmseg_utils import load_config_from_mmseg
from vis4d.struct import SemanticMasks

from .base import BaseDenseHeadConfig


class MMDecodeHeadConfig(BaseDenseHeadConfig):
    """Config for mmsegmentation decode heads."""

    name: str = "decode_head"
    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]


def results_from_mmseg(
    results: torch.Tensor, img_metas: List[MMDetMetaData], device: torch.device
) -> List[SemanticMasks]:
    """Convert mmsegmentation seg_pred to Vis4D format."""
    masks = []
    results = F.interpolate(
        results,
        size=img_metas[0]["pad_shape"][:2],  # type: ignore
        mode="bilinear",
    )
    results = results.argmax(dim=1)
    for result, img_meta in zip(results, img_metas):
        ori_h, ori_w = img_meta["ori_shape"][:2]  # type: ignore
        masks.append(
            SemanticMasks(
                torch.tensor(
                    result[:ori_h, :ori_w].unsqueeze(0), device=device
                ).byte()
            ).to_nhw_mask()
        )
    return masks


def get_mmseg_config(config: MMDecodeHeadConfig) -> MMConfig:
    """Convert a DecodeHead config to a mmseg readable config."""
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
    assert config.name in cfg, f"DecodeHead config not found: {config.name}"
    cfg = cfg[config.name]

    # convert decode head attributes
    assert config.category_mapping is not None
    cfg["num_classes"] = len(config.category_mapping)

    if config.model_kwargs:
        add_keyword_args(config, cfg)
    return cfg
