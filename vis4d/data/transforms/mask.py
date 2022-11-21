"""Segmentation/Instance Mask Transform."""
from __future__ import annotations

import torch

from vis4d.data.const import COMMON_KEYS

from .base import Transform


@Transform(
    in_keys=(COMMON_KEYS.boxes2d_classes, COMMON_KEYS.masks),
    out_keys=(COMMON_KEYS.segmentation_masks,),
)
def convert_ins_masks_to_seg_mask():
    """Merge all instance masks into a single segmentation map."""

    def _convert(classes, masks):
        cats = torch.as_tensor(classes, dtype=masks.dtype)
        target, _ = (masks * cats[:, None, None]).max(dim=0)
        target[masks.sum(0) > 1] = 255  # discard overlapping instances
        return target

    return _convert


@Transform(
    in_keys=(COMMON_KEYS.boxes2d_classes,),
    out_keys=(COMMON_KEYS.boxes2d_classes,),
)
def remap_categories(mapping: list[int]):
    """Remap classes using a mapping list.

    Args:
        mapping (List[int]): List of class ids, such that classes will be
            mapped to its location in the list.
    """

    def _remap(classes: torch.Tensor):
        for i in range(len(classes)):
            classes[i] = mapping.index(classes[i])
        return classes

    return _remap
