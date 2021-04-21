"""Tracking visualziations."""
from typing import List

import torch
from PIL import Image

from openmt.struct import Boxes2D

from .image import draw_bbox, imshow_bboxes
from .utils import BoxType, ImageType, preprocess_boxes, preprocess_image


def draw_sequence(
    frames: List[ImageType], boxes: List[BoxType]
) -> List[Image]:
    """Draw predictions of a complete sequence."""
    processed_frames = [preprocess_image(f) for f in frames]
    for frame, boxes2d in zip(processed_frames, boxes):
        box_list, col_list, _ = preprocess_boxes(boxes2d)
        for box, col in zip(box_list, col_list):
            draw_bbox(frame, box, col)
    return processed_frames


def visualize_matches(
    key_inputs: List[ImageType],
    ref_inputs: List[ImageType],
    key_boxes: List[Boxes2D],
    ref_boxes: List[Boxes2D],
) -> None:
    """Visualize paired bounding boxes successively for batched frame pairs."""
    for batch_i, (key_box, ref_box) in enumerate(zip(key_boxes, ref_boxes)):
        target = (
            key_box.track_ids.view(-1, 1) == ref_box.track_ids.view(1, -1)
        ).int()
        for key_i in range(target.shape[0]):
            if target[key_i].sum() > 0:
                ref_i = torch.argmax(target[key_i]).item()
                print("key view")
                imshow_bboxes(key_inputs[batch_i], key_box[key_i])
                print("ref view")
                imshow_bboxes(ref_inputs[batch_i], ref_box[ref_i])
