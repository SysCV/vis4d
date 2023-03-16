"""Segmentation/Instance Mask Transform."""
from __future__ import annotations

import torch

from vis4d.data.const import CommonKeys

from .base import Transform


@Transform(
    in_keys=(CommonKeys.boxes2d_classes, CommonKeys.instance_masks),
    out_keys=(CommonKeys.segmentation_masks,),
)
def convert_to_seg_masks():
    """Merge all instance masks into a single segmentation map."""

    def _convert(classes, masks):
        cats = torch.as_tensor(classes, dtype=masks.dtype)
        target, _ = (masks * cats[:, None, None]).max(dim=0)
        target[masks.sum(0) > 1] = 255  # discard overlapping instances
        return target.unsqueeze(0)

    return _convert


@Transform(
    in_keys=(CommonKeys.boxes2d_classes,),
    out_keys=(CommonKeys.boxes2d_classes,),
)
def remap_categories(mapping: list[int]):
    """Remap classes using a mapping list.

    Args:
        mapping (List[int]): List of class ids, such that classes will be
            mapped to its location in the list.
    """

    def _remap(classes: torch.Tensor):
        for i, cls in enumerate(classes):
            classes[i] = mapping.index(cls)
        return classes

    return _remap
