"""Utility functions for image processing operations."""

from __future__ import annotations

import numpy as np
import torch

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    ArrayLikeFloat,
    ArrayLikeInt,
    ArrayLikeUInt,
    NDArrayBool,
    NDArrayF32,
    NDArrayUI8,
)
from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import (
    boxes3d_in_image,
    boxes3d_to_corners,
    transform_boxes3d,
)
from vis4d.op.geometry.projection import project_points
from vis4d.op.geometry.transform import inverse_rigid_transform
from vis4d.vis.util import DEFAULT_COLOR_MAPPING


def _get_box_label(
    category: str | None,
    score: float | None,
    track_id: int | None,
) -> str:
    """Gets a unique string representation for a box definition.

    Args:
        category (str): The category name
        score (float): The confidence score
        track_id (int): The track id

    Returns:
        str: Label for this box of format
            'class_name, track_id, score%'
    """
    labels = []

    if category is not None:
        labels.append(category)
    if track_id is not None:
        labels.append(str(track_id))
    if score is not None:
        labels.append(f"{score * 100:.1f}%")
    return ", ".join(labels)


def _to_binary_mask(
    mask: NDArrayUI8, ignore_class: int = 255
) -> tuple[NDArrayUI8, NDArrayUI8]:
    """Converts a mask to binary masks.

    Args:
        mask (NDArrayUI8): The mask to convert with shape [H, W].
        ignore_class (int): The class id to ignore. Defaults to 255.

    Returns:
        NDArrayUI8: The binary masks with shape [N, H, W].
        NDArrayUI8: The class ids for each binary mask.
    """
    binary_masks = []
    class_ids = []
    for class_id in np.unique(mask):
        if class_id == ignore_class:
            continue
        binary_masks.append(mask == class_id)
        class_ids.append(class_id)
    return np.stack(binary_masks, axis=0), np.array(class_ids, dtype=np.uint8)


def preprocess_boxes(
    boxes: ArrayLikeFloat,
    scores: None | ArrayLikeFloat = None,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    color_palette: list[tuple[int, int, int]] = DEFAULT_COLOR_MAPPING,
    class_id_mapping: dict[int, str] | None = None,
    default_color: tuple[int, int, int] = (255, 0, 0),
    categories: None | list[str] = None,
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
        categories (None | list[str], optional): List of categories for each
            box.

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

        if categories is not None:
            category = categories[idx]
        elif class_id is not None:
            category = class_id_mapping.get(class_id, str(class_id))
        else:
            category = None

        labels_proc.append(_get_box_label(category, score, track_id))
    return boxes_proc, labels_proc, colors_proc


def preprocess_boxes3d(
    image_hw: tuple[int, int],
    boxes3d: ArrayLikeFloat,
    intrinsics: ArrayLikeFloat,
    extrinsics: ArrayLikeFloat | None = None,
    scores: None | ArrayLikeFloat = None,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    color_palette: list[tuple[int, int, int]] = DEFAULT_COLOR_MAPPING,
    class_id_mapping: dict[int, str] | None = None,
    default_color: tuple[int, int, int] = (255, 0, 0),
    axis_mode: AxisMode = AxisMode.OPENCV,
    categories: None | list[str] = None,
) -> tuple[
    list[tuple[float, float, float]],
    list[list[tuple[float, float, float]]],
    list[str],
    list[tuple[int, int, int]],
    list[int],
]:
    """Preprocesses bounding boxes.

    Converts the given predicted bounding boxes and class/track information
    into lists of centers, corners, labels, colors and track_ids.
    """
    if class_id_mapping is None:
        class_id_mapping = {}

    boxes3d = array_to_numpy(boxes3d, n_dims=2, dtype=np.float32)
    intrinsics = array_to_numpy(intrinsics, n_dims=2, dtype=np.float32)

    boxes3d = torch.from_numpy(boxes3d)
    intrinsics = torch.from_numpy(intrinsics)

    if axis_mode != AxisMode.OPENCV:
        assert (
            extrinsics is not None
        ), "extrinsics must be provided to move boxes to camera coordiante."
        extrinsics = array_to_numpy(extrinsics, n_dims=2, dtype=np.float32)
        extrinsics = torch.from_numpy(extrinsics)
        global_to_cam = inverse_rigid_transform(extrinsics)
        boxes3d_cam = transform_boxes3d(
            boxes3d,
            global_to_cam,
            source_axis_mode=AxisMode.ROS,
            target_axis_mode=AxisMode.OPENCV,
        )
    else:
        boxes3d_cam = boxes3d

    corners = boxes3d_to_corners(boxes3d_cam, axis_mode=AxisMode.OPENCV)

    mask = boxes3d_in_image(corners, intrinsics, image_hw)

    boxes3d_np = boxes3d.numpy()
    corners_np = corners.numpy()

    scores_np = array_to_numpy(scores, n_dims=1, dtype=np.float32)
    class_ids_np = array_to_numpy(class_ids, n_dims=1, dtype=np.int32)
    track_ids_np = array_to_numpy(track_ids, n_dims=1, dtype=np.int32)

    centers_proc: list[tuple[float, float, float]] = []
    corners_proc: list[list[tuple[float, float, float]]] = []
    colors_proc: list[tuple[int, int, int]] = []
    labels_proc: list[str] = []
    track_ids_proc: list[int] = []

    if len(mask) == 1:
        if not mask[0]:
            return (
                centers_proc,
                corners_proc,
                labels_proc,
                colors_proc,
                track_ids_proc,
            )
    else:
        boxes3d_np = boxes3d_np[mask]
        corners_np = corners_np[mask]
        scores_np = scores_np[mask] if scores_np is not None else None
        class_ids_np = class_ids_np[mask] if class_ids_np is not None else None
        track_ids_np = track_ids_np[mask] if track_ids_np is not None else None

    for idx in range(corners_np.shape[0]):
        class_id = None if class_ids_np is None else class_ids_np[idx].item()
        score = None if scores_np is None else scores_np[idx].item()
        track_id = None if track_ids_np is None else track_ids_np[idx].item()

        if track_id is not None:
            color = color_palette[track_id % len(color_palette)]
        elif class_id is not None:
            color = color_palette[class_id % len(color_palette)]
        else:
            color = default_color

        centers_proc.append(
            (
                boxes3d_np[idx][0].item(),
                boxes3d_np[idx][1].item(),
                boxes3d_np[idx][2].item(),
            )
        )
        corners_proc.append([tuple(pts) for pts in corners_np[idx].tolist()])
        colors_proc.append(color)

        if categories is not None:
            category = categories[idx]
        elif class_id is not None:
            category = class_id_mapping.get(class_id, str(class_id))
        else:
            category = None

        labels_proc.append(_get_box_label(category, score, track_id))
        if track_id is not None:
            track_ids_proc.append(track_id)
    return centers_proc, corners_proc, labels_proc, colors_proc, track_ids_proc


def preprocess_masks(
    masks: ArrayLikeUInt,
    class_ids: ArrayLikeInt | None = None,
    color_mapping: list[tuple[int, int, int]] = DEFAULT_COLOR_MAPPING,
) -> tuple[list[NDArrayBool], list[tuple[int, int, int]]]:
    """Preprocesses predicted semantic or instance segmentation masks.

    Args:
        masks (ArrayLikeUInt): Masks of shape [H, W] or [N, H, W]. If the
            masks are of shape [H, W], they are assumed to be semantic
            segmentation masks, i.e. each pixel contains the class id.
            If the masks are of shape [N, H, W], they are assumed to be
            the binary masks of N instances.
        class_ids (ArrayLikeInt, None):  An array with class ids for each mask
            shape [N]. If None, then the masks must be semantic segmentation
            masks and the class ids are extracted from the masks.
        color_mapping (list[tuple[int, int, int]]): Color mapping for
            each class.

    Returns:
        tuple[list[masks], list[colors]]: Returns a list with all masks of
            shape [H, W] as well as a list with the corresponding colors.

    Raises:
        ValueError: If the masks have an invalid shape.
    """
    masks_np = array_to_numpy(masks, n_dims=None, dtype=np.uint8)

    if len(masks_np.shape) == 2:
        masks_np, class_ids = _to_binary_mask(masks_np)
    elif len(masks_np.shape) == 3:
        if class_ids is not None:
            class_ids = array_to_numpy(class_ids, n_dims=1, dtype=np.int32)
    else:
        raise ValueError(
            f"Expected masks to have 2 or 3 dimensions, but got "
            f"{len(masks_np.shape)}"
        )

    masks_binary = masks_np.astype(bool)
    mask_list: list[NDArrayBool] = []
    color_list: list[tuple[int, int, int]] = []

    for idx in range(masks_binary.shape[0]):
        mask = masks_binary[idx, ...]

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

    return image_np.astype(np.uint8)


def get_intersection_point(
    point1: tuple[float, float, float],
    point2: tuple[float, float, float],
    camera_near_clip: float,
) -> tuple[float, float, float]:
    """Get point intersecting with camera near plane on line point1 -> point2.

    The line is defined by two points in camera coordinates and their depth.

    Args:
        point1 (tuple[float x 3]): First point in camera coordinates.
        point2 (tuple[float x 3]): Second point in camera coordinates
        camera_near_clip (float): camera_near_clip

    Returns:
        tuple[float, float, float]: The intersection point in camera
            coordiantes.
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

    return ((1 - k) * x1 + k * x2, (1 - k) * y1 + k * y2, camera_near_clip)


def project_point(
    point: tuple[float, float, float], intrinsics: NDArrayF32
) -> tuple[float, float]:
    """Project single point into the image plane."""
    projected_x, projected_y = (
        project_points(
            torch.from_numpy(np.array([point], dtype=np.float32)),
            torch.from_numpy(intrinsics),
        )
        .squeeze(0)
        .numpy()
        .tolist()
    )
    return projected_x, projected_y
