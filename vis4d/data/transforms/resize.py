"""Resize augmentation."""
import random
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from vis4d.op.box.util import transform_bbox
from vis4d.struct_to_revise.structures import DictStrAny

from ..datasets.base import DataKeys, DictData
from .base import BaseTransform


def get_resize_shape(
    original_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
    keep_ratio: bool = True,
    align_long_edge: bool = False,
) -> Tuple[int, int]:
    """Get shape for resize, considering keep_ratio."""
    h, w = original_shape
    new_h, new_w = new_shape
    if keep_ratio:
        if align_long_edge:
            long_edge, short_edge = max(new_shape), min(new_shape)
            scale_factor = min(long_edge / max(h, w), short_edge / min(h, w))
        else:
            scale_factor = min(new_w / w, new_h / h)
        new_h = int(h * scale_factor + 0.5)
        new_w = int(w * scale_factor + 0.5)
    return new_h, new_w


def transform_from_shapes(input_shape, resize_shape) -> torch.Tensor:
    """Generate 3x3 scaling matrix from two shapes."""
    h, w = input_shape
    new_h, new_w = resize_shape
    transform = torch.eye(3)
    transform[0, 0] = new_w / w
    transform[1, 1] = new_h / h
    return transform


def resize_tensor(
    inputs: torch.Tensor, resize_hw: Tuple[int, int], interpolation="bilinear"
) -> torch.Tensor:
    """Resize tensor of shape [N, C, H, W].

    Args:
        inputs (torch.Tensor): _description_
        resize_hw (Tuple[int, int]): _description_
        interpolation (str, optional): Interpolation method. One of ["nearest", "bilinear", "bicubic"]. Defaults to "bilinear".

    Returns:
        torch.Tensor: Resized tensor.
    """
    assert interpolation in ["nearest", "bilinear", "bicubic"]
    align_corners = None if interpolation == "nearest" else False
    output = F.interpolate(
        inputs, resize_hw, mode=interpolation, align_corners=align_corners
    )
    return output


def resize_boxes(
    boxes: torch.Tensor, input_shape: Tuple[int, int], shape: Tuple[int, int]
) -> torch.Tensor:
    """Resize 2D bounding boxes."""
    return transform_bbox(transform_from_shapes(input_shape, shape), boxes)


def resize_intrinsics(
    intrinsics: torch.Tensor,
    input_shape: Tuple[int, int],
    shape: Tuple[int, int],
) -> torch.Tensor:
    """Scale camera intrinsics when resizing."""
    return torch.matmul(transform_from_shapes(input_shape, shape), intrinsics)


def resize_masks(masks: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Resize masks."""
    if len(masks) == 0:  # handle empty masks
        return masks
    return (
        resize_tensor(
            masks.float().unsqueeze(0), shape, interpolation="nearest"
        )
        .type(masks.dtype)
        .squeeze(0)
    )


def get_target_shape(
    input_shape: Tuple[int, int],
    shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    keep_ratio: bool = False,
    multiscale_mode: str = "range",
    scale_range: Tuple[float, float] = (1.0, 1.0),
    align_long_edge: bool = False,
):
    """Generate possibly random target shape.


    Args:
        input_shape: Original image size in (H, W) format.
        shape: Image shape to be resized to in (H, W) format. In
        multiscale mode 'list', shape represents the list of possible
        shapes for resizing.
        keep_ratio: If aspect ratio of original image should be kept, the
        new shape will modified to fit the aspect ratio of the original image.
        multiscale_mode: one of [range, list],
        scale_range: Range of sampled image scales in range mode, e.g.
        (0.8, 1.2), indicating minimum of 0.8 * shape and maximum of
        1.2 * shape.
        align_long_edge: If keep_ratio is true, this option indicates if shape
        should be automatically aligned with the long edge of the original
        image, e.g. original shape=(100, 80), original shape=(100, 200) will
        yield (125, 100) as new shape. Default: False.
    """
    assert multiscale_mode in ["list", "range"]
    if multiscale_mode == "list":
        assert isinstance(
            shape, list
        ), "Specify shape as list when using multiscale mode list."
        assert len(shape) >= 1
    else:
        assert isinstance(
            shape, tuple
        ), "Specify shape as tuple when using multiscale mode range."
        assert (
            scale_range[0] <= scale_range[1]
        ), f"Invalid scale range: {scale_range[1]} < {scale_range[0]}"

    if multiscale_mode == "range":
        assert isinstance(shape, tuple)
        if scale_range[0] < scale_range[1]:  # do multi-scale
            w_scale = (
                random.uniform(0, 1) * (scale_range[1] - scale_range[0])
                + scale_range[0]
            )
            h_scale = (
                random.uniform(0, 1) * (scale_range[1] - scale_range[0])
                + scale_range[0]
            )
        else:
            h_scale = w_scale = 1.0

        shape = int(shape[0] * h_scale), int(shape[1] * w_scale)
    else:
        assert isinstance(shape, list)
        shape = random.choice(shape)

    shape = get_resize_shape(input_shape, shape, keep_ratio, align_long_edge)
    return shape


class Resize(BaseTransform):
    """Resize Augmentation."""

    def __init__(
        self,
        shape: Union[Tuple[int, int], List[Tuple[int, int]]],
        keep_ratio: bool = False,
        multiscale_mode: str = "range",
        scale_range: Tuple[float, float] = (1.0, 1.0),
        align_long_edge: bool = False,
        in_keys: Tuple[str, ...] = (
            DataKeys.images,
            DataKeys.boxes2d,
            DataKeys.intrinsics,
            DataKeys.masks,
        ),
    ):
        """Init."""
        super().__init__(in_keys)
        self.shape = shape
        self.keep_ratio = keep_ratio
        self.multiscale_mode = multiscale_mode
        self.scale_range = scale_range
        self.align_long_edge = align_long_edge

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate (possibly random) parameters for resize."""
        input_shape = data[DataKeys.images].shape[-2:]
        target_shape = get_target_shape(
            input_shape,
            self.shape,
            self.keep_ratio,
            self.multiscale_mode,
            self.scale_range,
            self.align_long_edge,
        )
        return dict(input_shape=input_shape, target_shape=target_shape)

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Apply resize augmentation."""
        input_shape, target_shape = (
            parameters["input_shape"],
            parameters["target_shape"],
        )
        data[DataKeys.metadata]["input_hw"] = target_shape
        for in_key in self.in_keys:
            if not in_key in data:
                continue

            if in_key == DataKeys.images:
                data[DataKeys.images] = resize_tensor(
                    data[DataKeys.images], target_shape
                )
            elif in_key == DataKeys.boxes2d:
                data[DataKeys.boxes2d] = resize_boxes(
                    data[DataKeys.boxes2d], input_shape, target_shape
                )
            elif in_key == DataKeys.intrinsics:
                data[DataKeys.intrinsics] = resize_intrinsics(
                    data[DataKeys.intrinsics], input_shape, target_shape
                )
            elif in_key == DataKeys.masks:
                data[DataKeys.masks] = resize_masks(
                    data[DataKeys.masks], target_shape
                )

        return data
