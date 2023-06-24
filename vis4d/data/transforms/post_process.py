"""Post process after transformation."""
from __future__ import annotations

import torch

from vis4d.common.typing import NDArrayF32, NDArrayI32
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import bbox_area, bbox_clip

from .base import Transform


@Transform(
    in_keys=[K.boxes2d, K.boxes2d_classes, K.boxes2d_track_ids, K.input_hw],
    out_keys=[K.boxes2d, K.boxes2d_classes, K.boxes2d_track_ids],
)
class PostProcessBoxes2D:
    """Post process after transformation."""

    def __init__(
        self, min_area: float = 7.0 * 7.0, clip_bboxes_to_image: bool = True
    ) -> None:
        """Creates an instance of the class.

        Args:
            min_area (float): Minimum area of the bounding box. Defaults to
                7.0 * 7.0.
            clip_bboxes_to_image (bool): Whether to clip the bounding boxes to
                the image size. Defaults to True.
        """
        self.min_area = min_area
        self.clip_bboxes_to_image = clip_bboxes_to_image

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        classes_list: list[NDArrayI32],
        track_ids_list: list[NDArrayI32 | None] | None,
        input_hw_list: list[tuple[int, int]],
    ) -> tuple[list[NDArrayF32], list[NDArrayI32], list[NDArrayI32 | None]]:
        """Resize 2D bounding boxes.

        Args:
            boxes (Tensor): The bounding boxes to be resized.
            scale_factor (tuple[float, float]): scaling factor.

        Returns:
            Tensor: Resized bounding boxes according to parameters in resize.
        """
        track_ids = (
            track_ids_list
            if track_ids_list is not None
            else [None] * len(boxes_list)
        )
        for i, (boxes, classes, track_ids_) in enumerate(
            zip(boxes_list, classes_list, track_ids)
        ):
            boxes_ = torch.from_numpy(boxes)
            if self.clip_bboxes_to_image:
                boxes_ = bbox_clip(boxes_, input_hw_list[i])

            keep = (bbox_area(boxes_) >= self.min_area).numpy()

            boxes_list[i] = boxes[keep]
            classes_list[i] = classes[keep]

            if track_ids_ is not None:
                track_ids_list[i] = track_ids_[keep]

        return boxes_list, classes_list, track_ids_list
