"""Utility functions for image processing operations."""
from __future__ import annotations

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    ArrayLikeBool,
    ArrayLikeFloat,
    ArrayLikeInt,
    NDArrayBool,
    NDArrayUI8,
)
from vis4d.vis.util import DEFAULT_COLOR_MAPPING

from .canvas import CanvasBackend


def _get_box_label(
    class_id: int | None,
    score: float | None,
    track_id: int | None,
    class_id_mapping: dict[int, str] | None = None,
) -> str:
    """Gets a unique string representation for a box definition.

    Args:
        class_id (int): The class id for this box
        score (float): The confidence score
        track_id (int): The track id
        class_id_mapping (dict[int,str]): Mapping of class_id to class name

    Returns:
        str: Label for this box of format
            'class_name, track_id, score%'
    """
    labels = []
    if class_id_mapping is None:
        class_id_mapping = {}

    if class_id is not None:
        labels.append(class_id_mapping.get(class_id, str(class_id)))
    if track_id is not None:
        labels.append(str(track_id))
    if score is not None:
        labels.append(f"{score * 100:.1f}%")
    return ", ".join(labels)


def preprocess_boxes(
    boxes: ArrayLikeFloat,
    scores: None | ArrayLikeFloat = None,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    color_palette: list[tuple[int, int, int]] = DEFAULT_COLOR_MAPPING,
    class_id_mapping: dict[int, str] | None = None,
    default_color: tuple[int, int, int] = (255, 0, 0),
) -> tuple[
    list[tuple[float, float, float, float]],
    list[str],
    list[tuple[int, int, int]],
]:
    """Preprocesses bounding boxes.

    Converts the given predicted bounding boxes and class/track information
    into lists of corners, labels and colors.

    Args:
        boxes (ArrayLikeFloat): Boxes of shape [N, 4] where N is the number of
                            boxes and the second channel consists of
                            (x1,y1,x2,y2) box coordinates.
        scores (ArrayLikeFloat): Scores for each box shape [N]
        class_ids (ArrayLikeInt): Class id for each box shape [N]
        track_ids (ArrayLikeInt): Track id for each box shape [N]
        color_palette (list[tuple[float, float, float]]): Color palette for
            each id.
        class_id_mapping(dict[int, str], optional): Mapping from class id
            to color tuple (0-255).
        default_color (tuple[int, int, int]): fallback color for boxes of no
            class or track id is given.

    Returns:
        boxes_proc (list[tuple[float, float, float, float]]): List of box
            corners.
        labels_proc (list[str]): List of labels.
        colors_proc (list[tuple[int, int, int]]): List of colors.
    """
    if class_id_mapping is None:
        class_id_mapping = {}

    boxes = array_to_numpy(boxes, n_dims=2, dtype=np.float32)

    scores_np = array_to_numpy(scores, n_dims=1, dtype=np.float32)
    class_ids_np = array_to_numpy(class_ids, n_dims=1, dtype=np.int32)
    track_ids_np = array_to_numpy(track_ids, n_dims=1, dtype=np.int32)

    boxes_proc: list[tuple[float, float, float, float]] = []
    colors_proc: list[tuple[int, int, int]] = []
    labels_proc: list[str] = []

    # Only one box provided
    if len(boxes.shape) == 1:
        # unsqueeze one dimension
        boxes = boxes.reshape(1, -1)

    for idx in range(boxes.shape[0]):
        class_id = None if class_ids_np is None else class_ids_np[idx].item()
        score = None if scores_np is None else scores_np[idx].item()
        track_id = None if track_ids_np is None else track_ids_np[idx].item()

        if track_id is not None:
            color = color_palette[track_id % len(color_palette)]
        elif class_id is not None:
            color = color_palette[class_id % len(color_palette)]
        else:
            color = default_color

        boxes_proc.append(
            (
                boxes[idx][0].item(),
                boxes[idx][1].item(),
                boxes[idx][2].item(),
                boxes[idx][3].item(),
            )
        )
        colors_proc.append(color)
        labels_proc.append(
            _get_box_label(class_id, score, track_id, class_id_mapping)
        )
    return boxes_proc, labels_proc, colors_proc


def preprocess_masks(
    masks: ArrayLikeBool,
    class_ids: ArrayLikeInt | None,
    color_mapping: list[tuple[int, int, int]] = DEFAULT_COLOR_MAPPING,
) -> tuple[list[NDArrayBool], list[tuple[int, int, int]]]:
    """Preprocesses predicted semantic masks.

    Args:
        masks (ArrayLikeBool): The semantic masks of shape [N, h, w].
        class_ids (ArrayLikeInt, None):  An array with class ids for each mask
            shape [N]
        color_mapping (list[tuple[int, int, int]]): Color mapping for
            each semantic class

    Returns:
        tuple[list[masks], list[colors]]: Returns a list with all masks of
            shape [h,w] as well as a list with the corresponding colors.
    """
    masks = array_to_numpy(masks, n_dims=3, dtype=np.bool_)
    class_ids = array_to_numpy(class_ids, n_dims=1, dtype=np.int32)

    mask_list: list[NDArrayBool] = []
    color_list: list[tuple[int, int, int]] = []

    for idx in range(masks.shape[0]):
        mask = masks[idx, ...]

        class_id = None if class_ids is None else class_ids[idx].item()
        if class_id is not None:
            color = color_mapping[class_id % len(color_mapping)]
        else:
            color = color_mapping[idx % len(color_mapping)]
        mask_list.append(mask)
        color_list.append(color)
    return mask_list, color_list


def preprocess_image(image: ArrayLike, mode: str = "RGB") -> NDArrayUI8:
    """Validate and convert input image.

    Args:
        image: CHW or HWC image (ArrayLike) with C = 3.
        mode: input channel format (e.g. BGR, HSV).

    Returns:
        np.array[uint8]: Processed image_np in RGB.
    """
    image_np = array_to_numpy(image, n_dims=3, dtype=np.float32)
    # Convert torch to numpy
    assert len(image_np.shape) == 3
    assert image_np.shape[0] == 3 or image_np.shape[-1] == 3

    # Convert torch to numpy convention
    if not image_np.shape[-1] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))

    # Convert image_np to [0, 255]
    min_val, max_val = (
        np.min(image_np, axis=(0, 1)),
        np.max(image_np, axis=(0, 1)),
    )
    image_np = image_np.astype(np.float32)
    image_np = (image_np - min_val) / (max_val - min_val) * 255.0

    if mode == "BGR":
        image_np = image_np[..., [2, 1, 0]]
        mode = "RGB"

    return image_np.astype(np.uint8)


def get_intersection_point(
    point1: tuple[float, float, float],
    point2: tuple[float, float, float],
    camera_near_clip: float,
) -> tuple[float, float]:
    """Get point intersecting with camera near plane on line point1 -> point2.

    The line is defined by two points in pixel coordinates and their depth.

    Args:
        point1 (tuple[float x 3]): First point in camera coordinates.
        point2 (tuple[float x 3]): Second point in camera coordinates
        camera_near_clip (float): camera_near_clip

    Returns:
        tuple[float x 2]: The intersection point in camera coordiantes.
    """
    c1, c2, c3 = 0, 0, camera_near_clip
    a1, a2, a3 = 0, 0, 1
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    k_up = abs(a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3))
    k_down = abs(a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2))
    if k_up > k_down:
        k = 1.0
    else:
        k = k_up / k_down
    return ((1 - k) * x1 + k * x1, (1 - k) * x2 + k * x2)


def draw_box3d(
    canvas: CanvasBackend,
    corners: tuple[tuple[float, float], ...],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw 3D bounding box on a given 2D canvas.

    Args:
        canvas (CanvasBackend): Current canvas to draw on.
        corners (tuple[tuple[float, float], ...]): Projected locations of the
            3D bounding box corners.
        label (str): Text label of the 3D box.
        color (tuple[int, int, int]): The box color.
    """
    assert len(corners) == 8, "A 3D box needs 8 corners."
    # Draw the sides
    for i in range(4):
        canvas.draw_line(corners[i], corners[i + 4], color)

    # Draw bottom (first 4 corners) and top (last 4 corners)
    canvas.draw_rotated_box(corners[:4], color)
    canvas.draw_rotated_box(corners[4:], color)

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners[:2], axis=0)
    center_bottom = np.mean(corners[:4], axis=0)
    canvas.draw_line(center_bottom, center_bottom_forward, color)

    # Draw label
    canvas.draw_text(corners[0], label, color)
