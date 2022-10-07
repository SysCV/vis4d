"""Segmentation/Instance Mask Transform."""
from typing import Tuple

import torch

from vis4d.data.datasets.base import COMMON_KEYS, DictData
from vis4d.struct_to_revise import DictStrAny

from .base import Transform


def convert_ins_masks_to_seg_mask(classes, masks):
    """Merge all instance masks into a single segmentation map
    with its corresponding categories."""

    cats = torch.as_tensor(classes, dtype=masks.dtype)
    target, _ = (masks * cats[:, None, None]).max(dim=0)
    target[masks.sum(0) > 1] = 255  # discard overlapping instances
    return target


class ConvertInsMasksToSegMask(Transform):
    """Convert instance masks to one segmentation map."""

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (
            COMMON_KEYS.boxes2d_classes,
            COMMON_KEYS.masks,
        ),
    ):
        """Init."""
        super().__init__(in_keys)

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate parameters (empty)."""
        return {}

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Convert masks."""
        data[COMMON_KEYS.segmentation_mask] = convert_ins_masks_to_seg_mask(
            data[self.in_keys[0]], data[self.in_keys[1]]
        )
        return data
