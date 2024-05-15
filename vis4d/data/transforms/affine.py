"""Affine transformation.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import math
import random
from typing import TypedDict

import numpy as np
import torch

from vis4d.common.imports import OPENCV_AVAILABLE
from vis4d.common.typing import NDArrayF32, NDArrayI64
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import bbox_clip, bbox_project

from .base import Transform
from .crop import _get_keep_mask

if OPENCV_AVAILABLE:
    import cv2
else:
    raise ImportError("Please install opencv-python to use this module.")


class AffineParam(TypedDict):
    """Parameters for Affine."""

    warp_matrix: NDArrayF32
    height: int
    width: int


def get_rotation_matrix(rotate_degrees: float) -> NDArrayF32:
    """Generate rotation matrix.

    Args:
        rotate_degrees (float): Rotation degrees.
    """
    radian = math.radians(rotate_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(radian), -np.sin(radian), 0.0],
            [np.sin(radian), np.cos(radian), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rotation_matrix


def get_scaling_matrix(scale_ratio: float) -> NDArrayF32:
    """Generate scaling matrix.

    Args:
        scale_ratio (float): Scale ratio.
    """
    scaling_matrix = np.array(
        [[scale_ratio, 0.0, 0.0], [0.0, scale_ratio, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return scaling_matrix


def get_shear_matrix(
    x_shear_degrees: float, y_shear_degrees: float
) -> NDArrayF32:
    """Generate shear matrix.

    Args:
        x_shear_degrees (float): X shear degrees.
        y_shear_degrees (float): Y shear degrees.
    """
    x_radian = math.radians(x_shear_degrees)
    y_radian = math.radians(y_shear_degrees)
    shear_matrix = np.array(
        [
            [1, np.tan(x_radian), 0.0],
            [np.tan(y_radian), 1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return shear_matrix


def get_translation_matrix(x_trans: float, y_trans: float) -> NDArrayF32:
    """Generate translation matrix.

    Args:
        x_trans (float): X translation.
        y_trans (float): Y translation.
    """
    translation_matrix = np.array(
        [[1, 0.0, x_trans], [0.0, 1, y_trans], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return translation_matrix


@Transform(K.input_hw, ["transforms.affine"])
class GenAffineParameters:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear, and scaling transforms.
    """

    def __init__(
        self,
        max_rotate_degree: float = 10.0,
        max_translate_ratio: float = 0.1,
        scaling_ratio_range: tuple[float, float] = (0.5, 1.5),
        max_shear_degree: float = 2.0,
        border: tuple[int, int] = (0, 0),
    ) -> None:
        """Creates an instance of the class.

        Args:
            max_rotate_degree (float): Maximum degrees of rotation transform.
                Defaults to 10.
            max_translate_ratio (float): Maximum ratio of translation.
                Defaults to 0.1.
            scaling_ratio_range (tuple[float]): Min and max ratio of
                scaling transform. Defaults to (0.5, 1.5).
            max_shear_degree (float): Maximum degrees of shear
                transform. Defaults to 2.
            border (tuple[int, int]): Distance from height and width sides of
                input image to adjust output shape. Only used in mosaic
                dataset. Defaults to (0, 0).
        """
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border

    def _get_random_homography_matrix(
        self, height: int, width: int
    ) -> NDArrayF32:
        """Generate random homography matrix."""
        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio_range[0], self.scaling_ratio_range[1]
        )
        scaling_matrix = get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(
            -self.max_shear_degree, self.max_shear_degree
        )
        y_degree = random.uniform(
            -self.max_shear_degree, self.max_shear_degree
        )
        shear_matrix = get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio)
            * width
        )
        trans_y = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio)
            * height
        )
        translate_matrix = get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
        )
        return warp_matrix

    def __call__(self, input_hw: list[tuple[int, int]]) -> list[AffineParam]:
        """Compute the parameters and put them in the data dict."""
        img_shape = input_hw[0]
        height = img_shape[0] + self.border[0] * 2
        width = img_shape[1] + self.border[1] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)
        return [
            AffineParam(warp_matrix=warp_matrix, height=height, width=width)
        ] * len(input_hw)


@Transform(
    [
        K.images,
        "transforms.affine.warp_matrix",
        "transforms.affine.height",
        "transforms.affine.width",
    ],
    [K.images, K.input_hw],
)
class AffineImages:
    """Affine Images."""

    def __init__(
        self,
        border_val: tuple[int, int, int] = (114, 114, 114),
        as_int: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            border_val (tuple[int, int, int]): Border padding values of 3
                channels. Defaults to (114, 114, 114).
            as_int (bool): Whether to convert the image to int. Defaults to
                False.
        """
        self.border_val = border_val
        self.as_int = as_int

    def __call__(
        self,
        images: list[NDArrayF32],
        warp_matrix_list: list[NDArrayF32],
        height_list: list[int],
        width_list: list[int],
    ) -> tuple[list[NDArrayF32], list[tuple[int, int]]]:
        """Affine a list of image of dimensions [N, H, W, C]."""
        input_hw_list = []
        for i, (image, warp_matrix, height, width) in enumerate(
            zip(images, warp_matrix_list, height_list, width_list)
        ):
            image = image[0].astype(np.uint8) if self.as_int else image[0]
            image = cv2.warpPerspective(  # pylint: disable=no-member, unsubscriptable-object, line-too-long
                image,
                warp_matrix,
                dsize=(width, height),
                borderValue=self.border_val,
            )[
                None, ...
            ].astype(
                np.float32
            )

            images[i] = image
            input_hw_list.append((height, width))
        return images, input_hw_list


@Transform(
    in_keys=[
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        "transforms.affine.warp_matrix",
        "transforms.affine.height",
        "transforms.affine.width",
    ],
    out_keys=[K.boxes2d, K.boxes2d_classes, K.boxes2d_track_ids],
)
class AffineBoxes2D:
    """Apply Affine to a list of 2D bounding boxes."""

    def __init__(self, bbox_clip_border: bool = True) -> None:
        """Creates an instance of the class.

        Args:
            bbox_clip_border (bool, optional): Whether to clip the objects
                outside the border of the image. In some dataset like MOT17,
                the gt bboxes are allowed to cross the border of images.
                Therefore, we don't need to clip the gt bboxes in these cases.
                Defaults to True.
        """
        self.bbox_clip_border = bbox_clip_border

    def __call__(
        self,
        boxes: list[NDArrayF32],
        classes: list[NDArrayI64],
        track_ids: list[NDArrayI64] | None,
        warp_matrix_list: list[NDArrayF32],
        height_list: list[int],
        width_list: list[int],
    ) -> tuple[list[NDArrayF32], list[NDArrayI64], list[NDArrayI64] | None]:
        """Apply Affine to 2D bounding boxes."""
        for i, (box, class_, warp_matrix, height, width) in enumerate(
            zip(
                boxes,
                classes,
                warp_matrix_list,
                height_list,
                width_list,
            )
        ):
            box_ = bbox_project(
                torch.from_numpy(box), torch.from_numpy(warp_matrix)
            )
            if self.bbox_clip_border:
                box_ = bbox_clip(box_, (height, width))
            boxes[i] = box_.numpy()

            keep_mask = _get_keep_mask(
                boxes[i], np.array([0, 0, width, height])
            )
            boxes[i] = boxes[i][keep_mask]
            classes[i] = class_[keep_mask]
            if track_ids is not None:
                track_ids[i] = track_ids[i][keep_mask]

        return boxes, classes, track_ids
