"""Resize augmentation."""
import random
from typing import Callable, List, Tuple, Union

import torch
import torch.nn.functional as F

from vis4d.data.datasets.base import COMMON_KEYS, MetaData
from vis4d.op.box.util import transform_bbox

from .base import Transform


def _get_resize_shape(
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


def _transform_from_shapes(
    input_shape: Tuple[int, int], resize_shape: Tuple[int, int]
) -> torch.Tensor:
    """Generate 3x3 scaling matrix from two shapes."""
    h, w = input_shape
    new_h, new_w = resize_shape
    transform = torch.eye(3)
    transform[0, 0] = new_w / w
    transform[1, 1] = new_h / h
    return transform


def _resize_tensor(
    inputs: torch.Tensor,
    shape: Tuple[int, int],
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """Resize Tensor of dimensions [N, C, H, W]."""
    assert interpolation in ["nearest", "bilinear", "bicubic"]
    align_corners = None if interpolation == "nearest" else False
    output = F.interpolate(
        inputs, shape, mode=interpolation, align_corners=align_corners
    )
    return output


@Transform(in_keys=[COMMON_KEYS.images, COMMON_KEYS.metadata])
def resize_image(
    shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    keep_ratio: bool = False,
    multiscale_mode: str = "range",
    scale_range: Tuple[float, float] = (1.0, 1.0),
    align_long_edge: bool = False,
    interpolation: str = "bilinear",
):
    """Resize tensor of shape [N, C, H, W].

    Args:
        shape (Union[Tuple[int, int], List[Tuple[int, int]]]): Image shape to be resized to in (H, W) format. In multiscale mode 'list', shape represents the list of possible shapes for resizing.
        keep_ratio (bool, optional): If aspect ratio of original image should be kept, the new shape will modified to fit the aspect ratio of the original image. Defaults to False.
        multiscale_mode (str, optional): one of [range, list]. Defaults to "range".
        scale_range (Tuple[float, float], optional): Range of sampled image scales in range mode, e.g. (0.8, 1.2), indicating minimum of 0.8 * shape and maximum of 1.2 * shape. Defaults to (1.0, 1.0).
        align_long_edge (bool, optional): If keep_ratio is true, this option indicates if shape should be automatically aligned with the long edge of the original image, e.g. original shape=(100, 80), original shape=(100, 200) will yield (125, 100) as new shape. Defaults to False.
        interpolation (str, optional): Interpolation method. One of ["nearest", "bilinear", "bicubic"]. Defaults to "bilinear".
    """

    def _resize(image: torch.Tensor, metadata: MetaData) -> torch.Tensor:
        im_shape = (image.size(2), image.size(3))
        tgt_shape = _get_target_shape(
            im_shape,
            shape,
            keep_ratio,
            multiscale_mode,
            scale_range,
            align_long_edge,
        )
        metadata["input_hw"] = tgt_shape
        return _resize_tensor(image, tgt_shape, interpolation=interpolation)

    return _resize


@Transform(
    in_keys=[COMMON_KEYS.boxes2d, COMMON_KEYS.metadata],
    out_keys=[COMMON_KEYS.boxes2d],
)
def resize_boxes2d():
    """Resize 2D bounding boxes."""

    def _resize(boxes: torch.Tensor, metadata: MetaData) -> torch.Tensor:
        input_shape, shape = metadata["original_hw"], metadata["input_hw"]
        return transform_bbox(
            _transform_from_shapes(input_shape, shape), boxes
        )

    return _resize


@Transform(
    in_keys=[COMMON_KEYS.intrinsics, COMMON_KEYS.metadata],
    out_keys=[COMMON_KEYS.intrinsics],
)
def resize_intrinsics():
    """Scale camera intrinsics when resizing."""

    def _resize(intrinsics: torch.Tensor, metadata: MetaData) -> torch.Tensor:
        input_shape, shape = metadata["original_hw"], metadata["input_hw"]
        return torch.matmul(
            _transform_from_shapes(input_shape, shape), intrinsics
        )

    return _resize


@Transform(
    in_keys=[COMMON_KEYS.masks, COMMON_KEYS.metadata],
    out_keys=[COMMON_KEYS.masks],
)
def resize_masks():
    """Resize masks."""

    def _resize(masks: torch.Tensor, metadata: MetaData) -> torch.Tensor:
        shape = metadata["input_hw"]
        if len(masks) == 0:  # handle empty masks
            return masks
        return (
            _resize_tensor(
                masks.float().unsqueeze(0), shape, interpolation="nearest"
            )
            .type(masks.dtype)
            .squeeze(0)
        )

    return _resize


def _get_target_shape(
    input_shape: Tuple[int, int],
    shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    keep_ratio: bool = False,
    multiscale_mode: str = "range",
    scale_range: Tuple[float, float] = (1.0, 1.0),
    align_long_edge: bool = False,
) -> Tuple[int, int]:
    """Generate possibly random target shape."""
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

    shape = _get_resize_shape(input_shape, shape, keep_ratio, align_long_edge)
    return shape
