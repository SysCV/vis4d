"""Utilities for visualization."""
import colorsys
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from openmt.struct import Boxes2D

ImageType = Union[torch.Tensor, np.ndarray]

BoxType = Union[Boxes2D, List[Boxes2D]]

ColorType = Union[
    Union[Tuple[int], str],
    List[Union[Tuple[int], str]],
    List[List[Union[Tuple[int], str]]],
]


def generate_colors(length: int) -> List[Tuple[int]]:
    """Generate a color palette of [length] colors."""
    brightness = 0.7
    hsv = [(i / length, 1, brightness) for i in range(length)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = (np.array(colors) * 255).astype(np.uint8).tolist()
    return [tuple(c) for c in colors]  # type: ignore


NUM_COLORS = 50
COLOR_PALETTE = generate_colors(NUM_COLORS)


def preprocess_boxes(
    boxes: BoxType, color_idx: int = 0
) -> Tuple[List[Tuple[float]], List[Tuple[int]], List[float]]:
    """Preprocess BoxType to boxes / colors for drawing."""
    if isinstance(boxes, list):
        result_box, result_color, result_score = [], [], []
        for i, b in enumerate(boxes):
            res_box, res_color, res_score = preprocess_boxes(b, i)
            result_box.extend(res_box)
            result_color.extend(res_color)
            result_score.extend(res_score)
        return result_box, result_color, result_score

    assert isinstance(boxes, Boxes2D)

    boxes_list = boxes.boxes[:, :4].cpu().numpy().tolist()
    scores = boxes.boxes[:, -1].cpu().numpy().tolist()

    if boxes.track_ids is not None:
        track_ids = boxes.track_ids.cpu().numpy()
        if len(track_ids.shape) > 1:
            track_ids = track_ids.squeeze(-1)
        draw_colors = [COLOR_PALETTE[t % NUM_COLORS] for t in track_ids]
    else:
        draw_colors = [
            COLOR_PALETTE[color_idx % NUM_COLORS]
            for _ in range(len(boxes_list))
        ]

    return boxes_list, draw_colors, scores


def preprocess_image(input_img: ImageType) -> Image.Image:
    """Validate and convert input image.

    Args:
        input_img: CHW or HWC image (ImageType) in RGB.

    Returns:
        PIL.Image.Image: Processed image.
    """
    assert len(input_img.shape) == 3
    assert input_img.shape[0] == 3 or input_img.shape[-1] == 3

    if isinstance(input_img, torch.Tensor):
        input_img = input_img.cpu().numpy()

    if not input_img.shape[-1] == 3:
        input_img = input_img.transpose(1, 2, 0)
    min_val, max_val = (
        np.min(input_img, axis=(0, 1)),
        np.max(input_img, axis=(0, 1)),
    )

    input_img = input_img.astype(np.float32)

    input_img = (input_img - min_val) / (max_val - min_val) * 255.0
    return Image.fromarray(input_img.astype(np.uint8))
