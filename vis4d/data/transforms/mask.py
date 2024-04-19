"""Segmentation/Instance Mask Transform."""

from __future__ import annotations

import numpy as np

from vis4d.common.typing import NDArrayI64, NDArrayUI8
from vis4d.data.const import CommonKeys as K

from .base import Transform


@Transform(
    in_keys=(K.boxes2d_classes, K.instance_masks),
    out_keys=K.seg_masks,
)
class ConvertInstanceMaskToSegMask:
    """Merge all instance masks into a single segmentation map."""

    def __call__(
        self, classes_list: list[NDArrayI64], masks_list: list[NDArrayUI8]
    ) -> list[NDArrayUI8]:
        """Execute conversion.

        Args:
            classes_list (list[NDArrayI64]): List of Array of class ids, shape
                [N,].
            masks_list (NDArrayU8): List of array of instance masks, shape
                [N, H, W].

        Returns:
            list[NDArrayU8]: List of Segmentation mask, shape [H, W].
        """
        seg_masks = []
        for classes, masks in zip(classes_list, masks_list):
            classes = np.asarray(classes, dtype=masks.dtype)
            target = np.max(masks * classes[:, None, None], axis=0)
            # discard overlapping instances
            target[np.sum(masks, axis=0) > 1] = 255

            seg_masks.append(target)
        return seg_masks


@Transform(
    in_keys=K.boxes2d_classes,
    out_keys=K.boxes2d_classes,
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

    def __call__(self, classes_list: list[NDArrayI64]) -> list[NDArrayI64]:
        """Execute remapping.

        Args:
            classes_list (list[NDArrayI64]): List of array of class ids, shape
                [N,].

        Returns:
            list[NDArrayI64]: List of array of remapped class ids, shape [N,].
        """
        for i, classes in enumerate(classes_list):
            for j, class_ in enumerate(classes):
                classes_list[i][j] = self.mapping.index(class_)
        return classes_list
