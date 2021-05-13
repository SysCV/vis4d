"""OpenMT Visualization tools for analysis and debugging."""
from typing import Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import BoxType, ImageType, preprocess_boxes, preprocess_image


def imshow(
    image: Union[Image.Image, ImageType], frame_id: Optional[int], folder: str
) -> None:  # pragma: no cover
    """Imshow method."""
    if not isinstance(image, Image.Image):
        image = preprocess_image(image)

    plt.imshow(np.asarray(image))
    plt.imsave(folder + "frame_" + str(frame_id) + ".png", np.asarray(image))
    # plt.show()


def imshow_bboxes(
    image: ImageType, boxes: BoxType, frame_id: Optional[int], folder: str
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image)
    box_list, color_list, _, trackid_list = preprocess_boxes(boxes)
    for box, col, trackid in zip(box_list, color_list, trackid_list):
        draw_bbox(image, box, col, trackid)

    imshow(image, frame_id, folder)


def draw_bbox(
    image: Image.Image, box: Tuple[float], color: Tuple[int], trackid: int
) -> None:
    """Draw 2D box onto image."""
    # print("draw_bbox:   ", box, "color: ", color, "")
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=color)
    if trackid is not None:
        font = ImageFont.load_default()
        draw.text(box[:2], str(trackid), (255, 255, 255), font=font)
