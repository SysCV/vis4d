"""Tracking visualziations."""
from typing import List, Sequence, Union

import torch
from PIL import Image

from vist.struct import Boxes2D

from .image import draw_image, imshow_bboxes
from .utils import BoxType, ImageType


def draw_sequence(
    frames: List[Union[ImageType, Image.Image]],
    boxes: Sequence[BoxType],
    mode: str = "RGB",
) -> List[Image.Image]:
    """Draw predictions of a complete sequence."""
    results = []
    for frame, box in zip(frames, boxes):
        results.append(draw_image(frame, boxes2d=box, mode=mode))
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
