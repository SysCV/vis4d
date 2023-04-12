"""Segmentation/Instance Mask Transform."""
from __future__ import annotations

import numpy as np

from vis4d.common.typing import NDArrayI32, NDArrayUI8
from vis4d.data.const import CommonKeys as K

from .base import Transform


@Transform(
    in_keys=(K.boxes2d_classes, K.instance_masks),
    out_keys=(K.seg_masks,),
)
class ConvertInstanceMaskToSegMask:
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
    in_keys=(K.boxes2d_classes,),
    out_keys=(K.boxes2d_classes,),
)
class RemappingCategories:
    """Remap classes using a mapping list."""

    def __init__(self, mapping: list[int]):
        """Initialize remapping.

        Args:
            mapping (List[int]): List of class ids, such that classes will be
                mapped to its location in the list.
        """
        self.mapping = mapping

    def __call__(self, classes: NDArrayI32) -> NDArrayI32:
        """Execute remapping.

        Args:
            classes (NDArrayI64): Array of class ids, shape [N,].

        Returns:
            NDArrayI64: Array of remapped class ids, shape [N,].
        """
        for i, cls in enumerate(classes):
            classes[i] = self.mapping.index(cls)
        return classes
