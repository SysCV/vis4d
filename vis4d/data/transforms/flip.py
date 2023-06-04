"""Horizontal flip augmentation."""
import numpy as np
import torch

from vis4d.common.typing import NDArrayF32, NDArrayUI8
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from .base import Transform


@Transform(K.images, K.images)
class FlipImages:
    """Flip a list of numpy image array of shape [N, H, W, C]."""

    def __init__(self, direction: str = "horizontal"):
        """Creates an instance of FlipImage.

        Args:
            direction (str, optional): Either vertical or horizontal.
                Defaults to "horizontal".

        Raises:
            ValueError: If direction is not horizontal or vertical.
        """
        if direction not in ["horizontal", "vertical"]:
            raise ValueError(f"Direction {self.direction} not known!")
        self.direction = direction

    def __call__(self, images: list[NDArrayF32]) -> list[NDArrayF32]:
        """Execute flipping op.

        Args:
            image (NDArrayF32): [N, H, W, C] array of image.

        Returns:
            NDArrayF32: [N, H, W, C] array of flipped image.
        """
        for i, image in enumerate(images):
            image_ = torch.from_numpy(image)
            if self.direction == "horizontal":
                images[i] = image_.flip(2).numpy()
            if self.direction == "vertical":
                images[i] = image_.flip(1).numpy()
        return images


@Transform(in_keys=(K.boxes2d, K.images), out_keys=(K.boxes2d,))
class FlipBoxes2D:
    """Flip a list of 2D bounding boxes."""

    def __init__(self, direction: str = "horizontal"):
        """Creates an instance of FlipBoxes2D.

        Args:
            direction (str, optional): Either vertical or horizontal.
                Defaults to "horizontal".

        Raises:
            ValueError: If direction is not horizontal or vertical.
        """
        if direction not in ["horizontal", "vertical"]:
            raise ValueError(f"Direction {self.direction} not known!")
        self.direction = direction

    def __call__(
        self, boxes: list[NDArrayF32], images: list[NDArrayF32]
    ) -> list[NDArrayF32]:
        """Execute flipping op.

        Args:
            boxes (list[NDArrayF32]): List of [M, 4] array of boxes.
            image (list[NDArrayF32]): List of [N, H, W, C] array of image.

        Returns:
            list[NDArrayF32]: List of [M, 4] array of flipped boxes.
        """
        for i, (boxes_, image) in enumerate(zip(boxes, images)):
            if self.direction == "horizontal":
                im_width = image.shape[2]
                tmp = im_width - boxes_[..., 2::4]
                boxes_[..., 2::4] = im_width - boxes_[..., 0::4]
                boxes_[..., 0::4] = tmp
            elif self.direction == "vertical":
                im_height = image.shape[1]
                tmp = im_height - boxes_[..., 3::4]
                boxes_[..., 3::4] = im_height - boxes_[..., 1::4]
                boxes_[..., 1::4] = tmp
            boxes[i] = boxes_
        return boxes


# TODO: Refactor this class
@Transform(K.seg_masks, K.seg_masks)
class FlipSegMasks:
    """Flip segmentation masks."""

    def __init__(self, direction: str = "horizontal"):
        """Creates an instance of FlipSemanticMasks.

        Args:
            direction (str, optional): Either vertical or horizontal.
                Defaults to "horizontal".
        """
        self.direction = direction

    def __call__(self, masks: NDArrayUI8) -> NDArrayUI8:
        """Execute flipping op.

        Args:
            masks (NDArrayUI8): [H, W] array of masks.

        Returns:
            NDArrayUI8: [H, W] array of flipped masks.
        """
        image_ = torch.from_numpy(masks)
        if self.direction == "horizontal":
            return image_.flip(1).numpy()
        if self.direction == "vertical":
            return image_.flip(0).numpy()
        raise ValueError(f"Direction {self.direction} not known!")


@Transform(K.instance_masks, K.instance_masks)
class FlipInstanceMasks:
    """Flip instance masks."""

    def __init__(self, direction: str = "horizontal"):
        """Creates an instance of FlipInstanceMasks.

        Args:
            direction (str, optional): Either vertical or horizontal.
                Defaults to "horizontal".

        Raises:
            ValueError: If direction is not horizontal or vertical.
        """
        if direction not in ["horizontal", "vertical"]:
            raise ValueError(f"Direction {self.direction} not known!")
        self.direction = direction

    def __call__(self, masks: list[NDArrayUI8]) -> NDArrayUI8:
        """Execute flipping op.

        Args:
            masks (list[NDArrayUI8]): List of [N, H, W] array of masks.

        Returns:
            list[NDArrayUI8]: List of [N, H, W] array of flipped masks.
        """
        for i, mask in enumerate(masks):
            mask_ = torch.from_numpy(mask)
            if self.direction == "horizontal":
                mask = mask_.flip(2).numpy()
            if self.direction == "vertical":
                mask = mask_.flip(1).numpy()
            masks[i] = mask
        return masks


def get_axis(direction: str, axis_mode: AxisMode) -> int:
    """Get axis number of certain direction given axis mode.

    Args:
        direction (str): One of horizontal, vertical and lateral.
        axis_mode (AxisMode): axis mode.

    Returns:
        int: Number of axis in certain direction.
    """
    if direction not in {"horizontal", "lateral", "vertical"}:
        raise ValueError(f"Direction {direction} not known!")
    coord_mapping = {
        AxisMode.ROS: {"horizontal": 0, "lateral": 1, "vertical": 2},
        AxisMode.OPENCV: {"horizontal": 0, "vertical": 1, "lateral": 2},
    }
    return coord_mapping[axis_mode][direction]


# TODO: Refactor this class
@Transform(in_keys=(K.boxes3d, K.axis_mode), out_keys=(K.boxes3d,))
class FlipBoxes3D:
    """Flip 3D bounding box array."""

    def __init__(self, direction: str = "horizontal"):
        """Creates an instance of FlipBoxes3D.

        Args:
            direction (str, optional): Either vertical or horizontal.
                Defaults to "horizontal".
        """
        self.direction = direction

    def __call__(self, boxes: NDArrayF32, axis_mode: AxisMode) -> NDArrayF32:
        """Execute flipping."""
        axis = get_axis(self.direction, axis_mode)
        angle_dir = "vertical" if self.direction == "horizontal" else "lateral"
        angles_axis = get_axis(angle_dir, axis_mode)
        boxes[:, axis] *= -1.0
        angles = matrix_to_euler_angles(
            quaternion_to_matrix(torch.from_numpy(boxes[:, 6:]))
        )
        angles[:, angles_axis] = np.pi - angles[:, angles_axis]
        boxes[:, 6:] = matrix_to_quaternion(
            euler_angles_to_matrix(angles)
        ).numpy()
        return boxes


# TODO: Refactor this class
@Transform(in_keys=(K.points3d, K.axis_mode), out_keys=(K.points3d,))
class FlipPoints3D:
    """Flip pointcloud array."""

    def __init__(self, direction: str = "horizontal"):
        """Creates an instance of FlipBoxes2D.

        Args:
            direction (str, optional): Either vertical or horizontal.
                Defaults to "horizontal".
        """
        self.direction = direction

    def __call__(
        self, points3d: NDArrayF32, axis_mode: AxisMode
    ) -> NDArrayF32:
        """Execute flipping."""
        points3d[:, get_axis(self.direction, axis_mode)] *= -1.0
        return points3d


# TODO: Refactor this class
@Transform(in_keys=(K.intrinsics, K.images), out_keys=(K.intrinsics,))
class FlipIntrinsics:
    """Modify intrinsics for image flip."""

    def __init__(self, direction: str = "horizontal"):
        """Creates an instance of FlipIntrinsics.

        Args:
            direction (str, optional): Either vertical or horizontal.
                Defaults to "horizontal".
        """
        self.direction = direction

    def __call__(
        self, intrinsics: NDArrayF32, image: NDArrayF32
    ) -> NDArrayF32:
        """Execute flipping."""
        if self.direction == "horizontal":
            center = image.shape[2] / 2
            intrinsics[0, 2] = center - intrinsics[0, 2] + center
            return intrinsics
        if self.direction == "vertical":
            center = image.shape[1] / 2
            intrinsics[1, 2] = center - intrinsics[1, 2] + center
            return intrinsics
        raise ValueError(f"Direction {self.direction} not known!")
