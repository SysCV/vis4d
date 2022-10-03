"""Horizontal flip augmentation."""
from typing import Tuple

import torch
import numpy as np
from vis4d.data.datasets.base import DataKeys, DictData
from vis4d.data.transforms.base import Transform
from vis4d.op.geometry.rotation import normalize_angle
from vis4d.data.transforms.base import Transform
from vis4d.struct_to_revise import DictStrAny


def hflip(tensor: torch.Tensor) -> torch.Tensor:
    """Flip a tensor of shape [N, C, H, W] horizontally."""
    return tensor.flip(-1)


def hflip_boxes2d(boxes: torch.Tensor, im_width: float) -> torch.Tensor:
    """Flip 2D bounding box tensor."""
    tmp = im_width - boxes[..., 2::4]
    boxes[..., 2::4] = im_width - boxes[..., 0::4]
    boxes[..., 0::4] = tmp
    return boxes


def hflip_boxes3d(boxes: torch.Tensor) -> torch.Tensor:
    """Flip 3D bounding box tensor."""
    boxes[:, 0] *= -1.0
    boxes[:, 7] = normalize_angle(np.pi - boxes[:, 7])
    return boxes


def hflip_points(points: torch.Tensor) -> torch.Tensor:
    """Flip pointcloud tensor."""
    points[:, 0] *= -1.0
    return points


def hflip_intrinsics(intrinsics: torch.Tensor, im_width: int) -> torch.Tensor:
    """Modify intrinsics for image flip."""
    center = im_width / 2
    intrinsics[0, 2] = center - intrinsics[0, 2] - center
    return intrinsics


class HorizontalFlip(Transform):
    """Horizontal flip augmentation class."""

    def __init__(
        self, in_keys: Tuple[str, ...] = (DataKeys.images, DataKeys.boxes2d)
    ):
        """Init."""
        super().__init__(in_keys)

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate flip parameters (empty)."""
        im_shape = data[DataKeys.images].shape[-2:]
        return dict(im_shape=im_shape)

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Apply horizontal flip."""
        im_shape = parameters["im_shape"]
        data[DataKeys.images] = hflip(data[DataKeys.images])
        data[DataKeys.boxes2d] = hflip_boxes2d(
            data[DataKeys.boxes2d], im_shape[1]
        )

        return data
