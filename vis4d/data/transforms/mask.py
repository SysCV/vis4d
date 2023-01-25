"""Segmentation/Instance Mask Transform."""
from __future__ import annotations

from typing import Callable

import torch

from vis4d.data.const import CommonKeys

from .base import Transform


@Transform(
    in_keys=(CommonKeys.boxes2d_classes, CommonKeys.masks),
    out_keys=(CommonKeys.segmentation_masks,),
)
def convert_to_seg_masks() -> Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor
]:
    """Merge all instance masks into a single segmentation map."""

    def _convert(classes, masks) -> torch.Tensor:
        cats = torch.as_tensor(classes, dtype=masks.dtype)
        target, _ = (masks * cats[:, None, None]).max(dim=0)
        target[masks.sum(0) > 1] = 255  # discard overlapping instances
        return target

    return _convert


@Transform(
    in_keys=(CommonKeys.boxes2d_classes,),
    out_keys=(CommonKeys.boxes2d_classes,),
)
def remap_categories(
    mapping: list[int],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Remap classes using a mapping list.

    Args:
        mapping (List[int]): List of class ids, such that classes will be
            mapped to its location in the list.
    """

    def _remap(classes: torch.Tensor) -> torch.Tensor:
        for i, cls in enumerate(classes):
            classes[i] = mapping.index(cls)
        return classes

    return _remap
