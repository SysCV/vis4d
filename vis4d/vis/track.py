"""Tracking visualziations."""
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image

from vis4d.struct import Boxes2D, Intrinsics, NDArrayF64

from .image import draw_image, imshow_bboxes
from .utils import Box3DType, BoxType, ImageType


def draw_sequence(
    frames: List[Union[ImageType, Image.Image]],
    boxes2d: Optional[Sequence[BoxType]] = None,
    boxes3d: Optional[Sequence[Box3DType]] = None,
    intrinsics: Optional[Sequence[Union[NDArrayF64, Intrinsics]]] = None,
    mode: str = "RGB",
) -> List[Image.Image]:
    """Draw predictions of a complete sequence."""
    results = []
    for i, frame in enumerate(frames):
        box2d = None
        if boxes2d is not None:
            box2d = boxes2d[i]
        box3d = None
        if boxes3d is not None:
            box3d = boxes3d[i]
        intr = None
        if intrinsics is not None:
            intr = intrinsics[i]
        results.append(
            draw_image(
                frame, boxes2d=box2d, boxes3d=box3d, intrinsics=intr, mode=mode
            )
        )
    return results


def visualize_matches(
    key_inputs: List[ImageType],
    ref_inputs: List[ImageType],
    key_boxes: List[Boxes2D],
    ref_boxes: List[Boxes2D],
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Visualize paired bounding boxes successively for batched frame pairs."""
    for batch_i, (key_box, ref_box) in enumerate(zip(key_boxes, ref_boxes)):
        target = (
            key_box.track_ids.view(-1, 1) == ref_box.track_ids.view(1, -1)
        ).int()
        for key_i in range(target.shape[0]):
            if target[key_i].sum() > 0:
                ref_i = torch.argmax(target[key_i]).item()
                print("key view")
                imshow_bboxes(key_inputs[batch_i], key_box[key_i], mode)
                print("ref view")
                imshow_bboxes(ref_inputs[batch_i], ref_box[ref_i], mode)
