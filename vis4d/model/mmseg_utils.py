"""Utilities for mmseg wrapper."""
import os
from typing import List, Union

import requests
import torch
import torch.nn.functional as F

from vis4d.struct import LabelInstances, NDArrayUI8, SemanticMasks

try:
    from mmcv import Config as MMConfig

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

MMResults = Union[List[NDArrayUI8], torch.Tensor]


def segmentations_from_mmseg(
    masks: torch.Tensor, device: torch.device
) -> List[SemanticMasks]:  # pragma: no cover
    """Convert mmsegmentation segmentations to Vis4D format."""
    return [
        SemanticMasks(mask).to_nhw_mask().to(device)
        for mask in F.softmax(masks, dim=1)
    ]


def results_from_mmseg(
    results: MMResults, device: torch.device
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


def load_config(path: str) -> MMConfig:
    """Load config either from file or from URL."""
    if os.path.exists(path):
        cfg = MMConfig.fromfile(path)
        if cfg.get("model"):
            cfg = cfg["model"]
    elif path.startswith("mmseg://"):
        ex = os.path.splitext(path)[1]
        cfg = MMConfig.fromstring(
            load_config_from_mmseg(path.split("mmseg://")[-1]), ex
        ).model
    else:
        raise FileNotFoundError(f"MMSegmentation config not found: {path}")
    return cfg
