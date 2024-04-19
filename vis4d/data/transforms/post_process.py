"""Post process after transformation."""

from __future__ import annotations

import torch

from vis4d.common.typing import NDArrayF32, NDArrayI64
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import bbox_area, bbox_clip

from .base import Transform


@Transform(
    in_keys=[
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        K.input_hw,
        K.boxes3d,
        K.boxes3d_classes,
        K.boxes3d_track_ids,
    ],
    out_keys=[
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        K.boxes3d,
        K.boxes3d_classes,
        K.boxes3d_track_ids,
    ],
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
        classes_list: list[NDArrayI64],
        track_ids_list: list[NDArrayI64] | None,
        input_hw_list: list[tuple[int, int]],
        boxes3d_list: list[NDArrayF32] | None,
        boxes3d_classes_list: list[NDArrayI64] | None,
        boxes3d_track_ids_list: list[NDArrayI64] | None,
    ) -> tuple[
        list[NDArrayF32],
        list[NDArrayI64],
        list[NDArrayI64] | None,
        list[NDArrayF32] | None,
        list[NDArrayI64] | None,
        list[NDArrayI64] | None,
    ]:
        """Post process according to boxes2D after transformation.

        Args:
            boxes_list (list[NDArrayF32]): The bounding boxes to be post
                processed.
            classes_list (list[NDArrayF32]): The classes of the bounding boxes.
            track_ids_list (list[NDArrayI64] | None): The track ids of the
                bounding boxes.
            input_hw_list (list[tuple[int, int]]): The height and width of the
                input image.
            boxes3d_list (list[NDArrayF32] | None): The 3D bounding boxes to be
                post processed.
            boxes3d_classes_list (list[NDArrayI64] | None): The classes of the
                3D bounding boxes.
            boxes3d_track_ids_list (list[NDArrayI64] | None): The track ids of
                the 3D bounding boxes.

        Returns:
            tuple[list[NDArrayF32], list[NDArrayI64], list[NDArrayI64] | None,
                list[NDArrayF32] | None, list[NDArrayI64] | None,
                list[NDArrayI64] | None]: The post processed results.
        """
        new_track_ids: list[NDArrayI64] | None = (
            [] if track_ids_list is not None else None
        )
        new_boxes3d: list[NDArrayF32] | None = (
            [] if boxes3d_list is not None else None
        )
        new_boxes3d_classes: list[NDArrayI64] | None = (
            [] if boxes3d_classes_list is not None else None
        )
        new_boxes3d_track_ids: list[NDArrayI64] | None = (
            [] if boxes3d_track_ids_list is not None else None
        )
        for i, (boxes, classes) in enumerate(zip(boxes_list, classes_list)):
            boxes_ = torch.from_numpy(boxes)
            if self.clip_bboxes_to_image:
                boxes_ = bbox_clip(boxes_, input_hw_list[i])

            keep = (bbox_area(boxes_) >= self.min_area).numpy()

            boxes_list[i] = boxes[keep]
            classes_list[i] = classes[keep]

            if track_ids_list is not None:
                assert new_track_ids is not None
                new_track_ids.append(track_ids_list[i][keep])

            if boxes3d_list is not None:
                assert new_boxes3d is not None
                new_boxes3d.append(boxes3d_list[i][keep])

            if boxes3d_classes_list is not None:
                assert new_boxes3d_classes is not None
                new_boxes3d_classes.append(boxes3d_classes_list[i][keep])

            if boxes3d_track_ids_list is not None:
                assert new_boxes3d_track_ids is not None
                new_boxes3d_track_ids.append(boxes3d_track_ids_list[i][keep])

        return (
            boxes_list,
            classes_list,
            new_track_ids,
            new_boxes3d,
            new_boxes3d_classes,
            new_boxes3d_track_ids,
        )


@Transform(in_keys=[K.boxes2d_track_ids], out_keys=[K.boxes2d_track_ids])
class RescaleTrackIDs:
    """Rescale track ids."""

    def __call__(self, track_ids_list: list[NDArrayI64]) -> list[NDArrayI64]:
        """Rescale the track ids.

        Args:
            track_ids_list (list[NDArrayI64]): The track ids to be
                rescaled.

        Returns:
            list[NDArrayI64]: The rescaled track ids.
        """
        track_ids_all: dict[int, int] = {}
        for track_ids in track_ids_list:
            for track_id in track_ids:
                if track_id not in track_ids_all:
                    track_ids_all[track_id] = len(track_ids_all)

        for track_ids in track_ids_list:
            for i, track_id in enumerate(track_ids):
                track_ids[i] = track_ids_all[track_id]

        return track_ids_list
