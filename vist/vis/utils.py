"""Utilities for visualization."""
import colorsys
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from vist.struct import Boxes2D, Boxes3D, NDArrayF64, NDArrayUI8

ImageType = Union[torch.Tensor, NDArrayUI8, NDArrayF64]

BoxType = Union[Boxes2D, List[Boxes2D]]
Box3DType = Union[Boxes3D, List[Boxes3D]]

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
    colors = (np.array(colors) * 255).astype(np.uint8).tolist()  # type: ignore
    s = np.random.get_state()
    np.random.seed(0)
    result = [tuple(colors[i]) for i in np.random.permutation(len(colors))]
    np.random.set_state(s)
    return result  # type: ignore


NUM_COLORS = 50
COLOR_PALETTE = generate_colors(NUM_COLORS)


def preprocess_boxes(
    boxes: Union[BoxType, Box3DType], color_idx: int = 0
) -> Tuple[List[List[float]], List[Tuple[int]], List[str]]:
    """Preprocess BoxType to boxes / colors / labels for drawing."""
    if isinstance(boxes, list):
        result_box, result_color, result_labels = [], [], []
        for i, b in enumerate(boxes):
            box, color, labels = preprocess_boxes(b, i)  # type: ignore
            result_box.extend(box)
            result_color.extend(color)
            result_labels.extend(labels)
        return result_box, result_color, result_labels

    assert isinstance(boxes, (Boxes2D, Boxes3D))

    if boxes.boxes.shape[-1] in [5, 8, 10]:  # 4, 7, 9 DoF boxes + score
        boxes_list = boxes.boxes[:, :-1].cpu().numpy().tolist()
        scores = boxes.boxes[:, -1].cpu().numpy().tolist()
    else:
        boxes_list = boxes.boxes.cpu().numpy().tolist()
        scores = [None for _ in range(len(boxes_list))]

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

        if s is not None:
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
        np.min(image, axis=(0, 1)),  # type: ignore
        np.max(image, axis=(0, 1)),  # type: ignore
    )

    image = image.astype(np.float32)

    image = (image - min_val) / (max_val - min_val) * 255.0

    if mode == "BGR":
        image = image[..., [2, 1, 0]]
        mode = "RGB"

    return Image.fromarray(image.astype(np.uint8), mode=mode).convert("RGB")


def box3d_to_corners(box3d: List[float]) -> NDArrayF64:
    """Convert Boxes3D style box to its respective corner points."""
    x_loc, y_loc, z_loc = box3d[:3]
    h, w, l = box3d[3:6]
    if len(box3d) < 9:  # if 7 DoF box type
        rx, ry, rz = 0.0, box3d[6], 0.0
    else:
        rx, ry, rz = box3d[6], box3d[7], box3d[8]

    x_corners = np.array(
        [
            l / 2.0,
            l / 2.0,
            -l / 2.0,
            -l / 2.0,
            l / 2.0,
            l / 2.0,
            -l / 2.0,
            -l / 2.0,
        ],
        dtype=np.float64,
    )
    z_corners = np.array(
        [
            w / 2.0,
            -w / 2.0,
            -w / 2.0,
            w / 2.0,
            w / 2.0,
            -w / 2.0,
            -w / 2.0,
            w / 2.0,
        ],
        dtype=np.float64,
    )
    y_corners = np.zeros((8,), dtype=np.float64)
    y_corners[0:4] = h / 2
    y_corners[4:8] = -h / 2

    rot = R.from_euler("xyz", np.array([rx, ry, rz])).as_matrix()
    temp_corners = np.concatenate(
        (
            x_corners.reshape(8, 1),
            y_corners.reshape(8, 1),
            z_corners.reshape(8, 1),
        ),
        axis=1,
    )  # type: ignore
    corners = np.matmul(temp_corners, rot.T)
    corners[:, 0], corners[:, 1], corners[:, 2] = (
        corners[:, 0] + x_loc,
        corners[:, 1] + y_loc,
        corners[:, 2] + z_loc,
    )
    return corners  # type: ignore
