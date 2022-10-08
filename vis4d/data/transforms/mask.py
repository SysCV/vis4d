"""Segmentation/Instance Mask Transform."""

import torch

from vis4d.data.datasets.base import COMMON_KEYS

from .base import Transform


@Transform(
    in_keys=[COMMON_KEYS.boxes2d_classes, COMMON_KEYS.masks],
    out_keys=[COMMON_KEYS.segmentation_mask],
)
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
