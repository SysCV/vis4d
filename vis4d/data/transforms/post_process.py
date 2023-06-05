"""Post process after transformation."""
from __future__ import annotations

from typing import TypedDict

import torch

from vis4d.common.typing import NDArrayF32, NDArrayI32
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import bbox_area, bbox_clip

from .base import Transform


class PostProcBoxes2DParam(TypedDict):
    """Parameters for Resize."""

    valid_indices: list[int]


@Transform(
    [K.boxes2d, K.boxes2d_classes, K.boxes2d_track_ids, K.input_hw],
    [
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        "tansforms.post_process_boxes2d",
    ],
)
class PostProcessBoxes2d:
    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        classes_list: list[NDArrayI32],
        track_ids_list: list[NDArrayI32],
        input_hw_list: list[tuple[int, int]],
        min_area: float = 7.0 * 7.0,
        clip_bboxes_to_image: bool = True,
    ) -> NDArrayF32:
        """Resize 2D bounding boxes.

        Args:
            boxes (Tensor): The bounding boxes to be resized.
            scale_factor (tuple[float, float]): scaling factor.

        Returns:
            Tensor: Resized bounding boxes according to parameters in resize.
        """
        transformed_boxes_list = []
        transformed_classes_list = []
        transformed_track_ids_list = []
        transforms_params = []

        for i, boxes in enumerate(boxes_list):
            boxes_ = torch.from_numpy(boxes)
            if clip_bboxes_to_image:
                boxes_ = bbox_clip(boxes_, input_hw_list[i])

            keep = bbox_area(boxes_) >= min_area
            boxes_ = boxes_[keep]
            classes_ = classes_list[i][keep.numpy()]
            track_ids_ = track_ids_list[i][keep.numpy()]

            transformed_boxes_list.append(boxes_.numpy())
            transformed_classes_list.append(classes_)
            transformed_track_ids_list.append(track_ids_)

            transforms_params.append(
                PostProcBoxes2DParam(valid_indices=keep.numpy())
            )

        return (
            transformed_boxes_list,
            transformed_classes_list,
            transformed_track_ids_list,
            transforms_params,
        )
