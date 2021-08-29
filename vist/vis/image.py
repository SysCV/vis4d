"""VisT Visualization tools for analysis and debugging."""
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scalabel.label.utils import project_points_to_image

from vist.struct import Intrinsics, NDArrayF64

from .utils import (
    Box3DType,
    BoxType,
    ImageType,
    box3d_to_corners,
    preprocess_boxes,
    preprocess_image,
)


def imshow(
    image: Union[Image.Image, ImageType], mode: str = "BGR"
) -> None:  # pragma: no cover
    """Imshow method.

    Args:
        image: PIL Image or ImageType (i.e. numpy array, torch.Tensor)
        mode: Image channel format, will be used to convert ImageType to
        an RGB PIL Image. Not necessary if 'image' is an RGB PIL Image.
    """
    if not isinstance(image, Image.Image):
        image = preprocess_image(image, mode)
    plt.imshow(np.asarray(image))
    plt.show()
    # image.save("visualization/" + "frame_" + str(frame_id).zfill(4) + ".png")


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
    frame_id: int = 000,
) -> None:
    image = preprocess_image(image, mode)
    box_list, color_list, label_list = preprocess_boxes(boxes)
    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox(image, box, col, label)

    if not isinstance(image, Image.Image):
        image = preprocess_image(image, mode)
    image.save("visualization/" + "frame_" + str(frame_id).zfill(4) + ".png")


def imshow_bboxes3d(
    image: ImageType,
    boxes: Box3DType,
    intrinsics: Union[NDArrayF64, Intrinsics],
    mode: str = "BGR",
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image, mode)
    box_list, color_list, label_list = preprocess_boxes(boxes)
    if isinstance(intrinsics, Intrinsics):
        intrinsic_matrix = intrinsics.tensor.cpu().numpy()  # type: NDArrayF64
    elif isinstance(intrinsics, np.ndarray):
        intrinsic_matrix = intrinsics
    else:
        raise ValueError(f"Invalid type for intrinsics: {type(intrinsics)}")

    assert intrinsic_matrix.shape == (
        3,
        3,
    ), f"Intrinsics must be of shape 3x3, got {intrinsic_matrix.shape}"

    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox3d(image, box, intrinsic_matrix, col, label)

    imshow(image)


def draw_bbox(
    image: Image.Image,
    box: List[float],
    color: Tuple[int],
    label: Optional[str] = None,
) -> None:
    """Draw 2D box onto image."""
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=color)
    if label is not None:
        font = ImageFont.load_default()
        draw.text(box[:2], label, (255, 255, 255), font=font)


def draw_bbox3d(
    image: Image.Image,
    box3d: List[float],
    intrinsics: NDArrayF64,
    color: Tuple[int],
    label: Optional[str] = None,
) -> None:  # pragma: no cover
    """Draw 3D box onto image."""
    draw = ImageDraw.Draw(image)
    corners = project_points_to_image(box3d_to_corners(box3d), intrinsics)

    def draw_rect(selected_corners: NDArrayF64) -> None:
        prev = selected_corners[-1]
        for corner in selected_corners:
            draw.line((tuple(prev), tuple(corner)), fill=color)
            prev = corner

    # Draw the sides
    for i in range(4):
        draw.line((tuple(corners[i]), tuple(corners[i + 4])), fill=color)

    # Draw bottom (first 4 corners) and top (last 4 corners)
    draw_rect(corners[:4])
    draw_rect(corners[4:])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners[:2], axis=0)
    center_bottom = np.mean(corners[:4], axis=0)
    draw.line((tuple(center_bottom), tuple(center_bottom_forward)), fill=color)

    if label is not None:
        font = ImageFont.load_default()
        center_top_forward = tuple(np.mean(corners[2:4], axis=0))
        draw.text(center_top_forward, label, (255, 255, 255), font=font)
