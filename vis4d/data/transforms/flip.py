"""Horizontal flip augmentation."""
import numpy as np
import torch

from vis4d.data.datasets.base import COMMON_KEYS
from vis4d.op.geometry.rotation import normalize_angle

from .base import Transform


@Transform()
def image_flip(direction: str = "horizontal"):
    """Flip a tensor of shape [N, C, H, W] horizontally."""

    def _flip(tensor: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            return tensor.flip(-1)
        elif direction == "vertical":
            return tensor.flip(-2)
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


@Transform(
    in_keys=(COMMON_KEYS.boxes2d, COMMON_KEYS.images),
    out_keys=(COMMON_KEYS.boxes2d,),
)
def boxes2d_flip(direction: str = "horizontal"):
    """Flip 2D bounding box tensor."""

    def _flip(boxes: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            im_width = image.size(3)
            tmp = im_width - boxes[..., 2::4]
            boxes[..., 2::4] = im_width - boxes[..., 0::4]
            boxes[..., 0::4] = tmp
            return boxes
        elif direction == "vertical":
            im_height = image.size(2)
            tmp = im_height - boxes[..., 3::4]
            boxes[..., 3::4] = im_height - boxes[..., 1::4]
            boxes[..., 1::4] = tmp
            return boxes
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


@Transform(in_keys=(COMMON_KEYS.boxes3d,), out_keys=(COMMON_KEYS.boxes3d,))
def boxes3d_flip(direction: str = "horizontal"):
    """Flip 3D bounding box tensor."""

    def _flip(boxes: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            boxes[:, 0] *= -1.0
            boxes[:, 7] = normalize_angle(np.pi - boxes[:, 7])
            return boxes
        elif direction == "vertical":
            boxes[:, 1] *= -1.0
            return boxes
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


@Transform(in_keys=(COMMON_KEYS.points3d,), out_keys=(COMMON_KEYS.points3d,))
def points3d_flip(direction: str = "horizontal"):
    """Flip pointcloud tensor."""

    def _flip(points3d: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            points3d[:, 0] *= -1.0
            return points
        elif direction == "vertical":
            points3d[:, 1] *= -1.0
            return points3d
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


@Transform(
    in_keys=(COMMON_KEYS.intrinsics,), out_keys=(COMMON_KEYS.intrinsics,)
)
def intrinsics_flip(direction: str = "horizontal"):
    """Modify intrinsics for image flip."""

    def _flip(intrinsics: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            center = image.size(3) / 2
            intrinsics[0, 2] = center - intrinsics[0, 2] - center
            return intrinsics
        elif direction == "vertical":
            center = image.size(2) / 2
            intrinsics[1, 2] = center - intrinsics[1, 2] - center
            return intrinsics
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip
