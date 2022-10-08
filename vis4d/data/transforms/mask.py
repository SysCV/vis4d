"""Segmentation/Instance Mask Transform."""
from typing import Tuple

import torch

from vis4d.data.datasets.base import DataKeys

from .base import Transform


@Transform(in_keys=[DataKeys.boxes2d_classes, DataKeys.masks], out_keys=[DataKeys.segmentation_mask])
def convert_ins_masks_to_seg_mask():
    """Convert instance masks to one segmentation map."""
    def _convert(classes, masks):
        """Merge all instance masks into a single segmentation map
        with its corresponding categories."""

        cats = torch.as_tensor(classes, dtype=masks.dtype)
        target, _ = (masks * cats[:, None, None]).max(dim=0)
        target[masks.sum(0) > 1] = 255  # discard overlapping instances
        return target
    return _convert
