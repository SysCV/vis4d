"""OpenMT Visualization tools for analysis and debugging."""
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import BoxType, ImageType, preprocess_boxes, preprocess_image


def imshow(
    image: Union[Image.Image, ImageType], mode: str = "BGR", frame_id: int = 0
) -> None:  # pragma: no cover  # pylint: disable=line-too-long
    """Imshow method.

    Args:
        image: PIL Image or ImageType (i.e. numpy array, torch.Tensor)
        mode: Image channel format, will be used to convert ImageType to
        an RGB PIL Image. Not necessary if 'image' is an RGB PIL Image.
    """
    if not isinstance(image, Image.Image):
        image = preprocess_image(image, mode)
    # plt.imshow(np.asarray(image))
    # plt.show()
    image.save("visualization/" + "frame_" + str(frame_id).zfill(4) + ".png")


def imshow_bboxes(
    image: ImageType, boxes: BoxType, mode: str = "BGR", frame_id: int = 0
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image, mode)
    box_list, color_list, label_list = preprocess_boxes(boxes)
    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox(image, box, col, label)

    imshow(image, frame_id=frame_id)


def imsave_bboxes(
    image: ImageType,
    boxes: BoxType,
    mode: str = "BGR",
    frame_id=000,
) -> None:
    image = preprocess_image(image, mode)
    box_list, color_list, label_list = preprocess_boxes(boxes)
    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox(image, box, col, label)

    if not isinstance(image, Image.Image):
        image = preprocess_image(image, mode)
    image.save("visualization/" + "frame_" + str(frame_id).zfill(4) + ".png")


def draw_bbox(
    image: Image.Image,
    box: Tuple[float],
    color: Tuple[int],
    label: Optional[str] = None,
) -> None:
    """Draw 2D box onto image."""
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=color)
    if label is not None:
        font = ImageFont.load_default()
        draw.text(box[:2], label, (255, 255, 255), font=font)
