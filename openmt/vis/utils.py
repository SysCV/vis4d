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
    s = np.random.get_state()
    np.random.seed(0)
    result = [tuple(colors[i]) for i in np.random.permutation(len(colors))]
    np.random.set_state(s)
    return result  # type: ignore


NUM_COLORS = 50
COLOR_PALETTE = generate_colors(NUM_COLORS)


def preprocess_boxes(
    boxes: BoxType, color_idx: int = 0
) -> Tuple[List[Tuple[float]], List[Tuple[int]], List[str]]:
    """Preprocess BoxType to boxes / colors / labels for drawing."""
    if isinstance(boxes, list):
        result_box, result_color, result_labels = [], [], []
        for i, b in enumerate(boxes):
            res_box, res_color, res_labels = preprocess_boxes(b, i)
            result_box.extend(res_box)
            result_color.extend(res_color)
            result_labels.extend(res_labels)
        return result_box, result_color, result_labels

    assert isinstance(boxes, Boxes2D)

    boxes_list = boxes.boxes[:, :4].cpu().detach().numpy().tolist()
    scores = boxes.boxes[:, -1].cpu().detach().numpy().tolist()

    if boxes.track_ids is not None:
        track_ids = boxes.track_ids.cpu().numpy()
        if len(track_ids.shape) > 1:
            track_ids = track_ids.squeeze(-1)
    else:
        track_ids = [None for _ in range(len(boxes_list))]

    if boxes.class_ids is not None:
        class_ids = boxes.class_ids.cpu().numpy()
    else:
        class_ids = [None for _ in range(len(boxes_list))]

    labels, draw_colors = [], []
    for s, t, c in zip(scores, track_ids, class_ids):
        if t is not None:
            draw_color = COLOR_PALETTE[int(t) % NUM_COLORS]
        elif c is not None:
            draw_color = COLOR_PALETTE[int(c) % NUM_COLORS]
        else:
            draw_color = COLOR_PALETTE[color_idx % NUM_COLORS]

        label = ""
        if t is not None:
            label += str(int(t)) + ","
        if c is not None:
            str_c = str(int(c))
            if boxes.metadata is not None:
                str_c = boxes.metadata[str_c]  # type: ignore
            label += str_c + ","

        label += "{:.1f}%".format(s * 100)
        labels.append(label)
        draw_colors.append(draw_color)

    return boxes_list, draw_colors, labels


def preprocess_image(image: ImageType, mode: str = "BGR") -> Image.Image:
    """Validate and convert input image.

    Args:
        image: CHW or HWC image (ImageType) with C = 3.
        mode: input channel format (e.g. BGR, HSV). More info
        at https://pillow.readthedocs.io/en/stable/handbook/concepts.html

    Returns:
        PIL.Image.Image: Processed image in RGB.
    """
    assert len(image.shape) == 3
    assert image.shape[0] == 3 or image.shape[-1] == 3

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if not image.shape[-1] == 3:
        image = image.transpose(1, 2, 0)
    min_val, max_val = (
        np.min(image, axis=(0, 1)),
        np.max(image, axis=(0, 1)),
    )

    image = image.astype(np.float32)

    image = (image - min_val) / (max_val - min_val) * 255.0

    if mode == "BGR":
        image = image[..., [2, 1, 0]]
        mode = "RGB"

    return Image.fromarray(
        image.astype(np.uint8), mode=mode
    )  # .convert("RGB")
