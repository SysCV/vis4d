"""Segmentation/Instance Mask Transform."""
from __future__ import annotations

import numpy as np

from vis4d.common.typing import NDArrayI32, NDArrayUI8
from vis4d.data.const import CommonKeys

from .base import Transform


@Transform(
    in_keys=(CommonKeys.boxes2d_classes, CommonKeys.instance_masks),
    out_keys=(CommonKeys.segmentation_masks,),
)
class ConvertInstanceMaskToSegmentationMask:
    """Merge all instance masks into a single segmentation map."""

    def __call__(self, classes: NDArrayI32, masks: NDArrayUI8) -> NDArrayUI8:
        """Execute conversion.

        Args:
            classes (NDArrayI64): Array of class ids, shape [N,].
            masks (NDArrayU8): Array of instance masks, shape [N, H, W].

        Returns:
            NDArrayU8: Segmentation mask, shape [H, W].
        """
        classes = np.asarray(classes, dtype=masks.dtype)
        target = np.max(masks * classes[:, None, None], axis=0)
        # discard overlapping instances
        target[np.sum(masks, axis=0) > 1] = 255
        return target


@Transform(
    in_keys=(CommonKeys.boxes2d_classes,),
    out_keys=(CommonKeys.boxes2d_classes,),
)
class RemappingCategories:
    def __init__(self, mapping: list[int]):
        """Remap classes using a mapping list.

        Args:
            mapping (List[int]): List of class ids, such that classes will be
                mapped to its location in the list.
        """
        self.mapping = mapping

    def __call__(self, classes: NDArrayI32) -> NDArrayI32:
        for i, cls in enumerate(classes):
            classes[i] = self.mapping.index(cls)
        return classes
