"""OpenMT Visualization tools for analysis and debugging."""
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from .utils import BoxType, ImageType, preprocess_boxes, preprocess_image


def imshow(image: Union[Image.Image, ImageType]) -> None:  # pragma: no cover
    """Imshow method."""
    if not isinstance(image, Image.Image):
        image = preprocess_image(image)
    plt.imshow(np.asarray(image))
    plt.show()


def imshow_bboxes(
    image: ImageType, boxes: BoxType
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image)
    box_list, color_list, _ = preprocess_boxes(boxes)
    for box, col in zip(box_list, color_list):
        draw_bbox(image, box, col)

    imshow(image)


def draw_bbox(
    image: Image.Image, box: Tuple[float], color: Tuple[int]
) -> None:
    """Draw 2D box onto image."""
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=color)
