"""Utilities for mmseg wrapper."""

from typing import List, Union

import requests
import torch
import torch.nn.functional as F

from vis4d.model.mmdet_utils import MMDetMetaData
from vis4d.struct import LabelInstances, NDArrayUI8, SemanticMasks

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
