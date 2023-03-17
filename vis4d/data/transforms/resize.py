"""Resize transformation."""
from __future__ import annotations

import random
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as CK
from vis4d.op.box.box2d import transform_bbox

from .base import Transform


class ResizeParam(TypedDict):
    """Parameters for Resize."""

    target_shape: tuple[int, int]
    scale_factor: tuple[float, float]
    interpolation: str


@Transform(CK.images, "transforms.resize")
class GenerateResizeParameters:
    """Generate the parameters for a resize operation."""

    def __init__(
        self,
        shape: tuple[int, int] | list[tuple[int, int]],
        keep_ratio: bool = False,
        multiscale_mode: str = "range",
        scale_range: tuple[float, float] = (1.0, 1.0),
        align_long_edge: bool = False,
        interpolation: str = "bilinear",
    ) -> None:
        """Creates an instance of the class.

        Args:
            shape (tuple[int, int] | list[tuple[int, int]]): Image shape to
                be resized to in (H, W) format. In multiscale mode 'list',
                shape represents the list of possible shapes for resizing.
            keep_ratio (bool, optional): If aspect ratio of the original image
                should be kept, the new shape will modified to fit the aspect
                ratio of the original image. Defaults to False.
            multiscale_mode (str, optional): one of [range, list]. Defaults to
                "range".
            scale_range (tuple[float, float], optional): Range of sampled image
                scales in range mode, e.g. (0.8, 1.2), indicating minimum of
                0.8 * shape and maximum of 1.2 * shape. Defaults to (1.0, 1.0).
            align_long_edge (bool, optional): If keep_ratio=true, this option
                indicates if shape should be automatically aligned with the
                long edge of the original image, e.g. original shape=(100, 80),
                shape to be resized=(100, 200) will yield (125, 100) as new
                shape. Defaults to False.
            interpolation (str, optional): Interpolation method. One of
                ["nearest", "bilinear", "bicubic"]. Defaults to "bilinear".
        """
        self.shape = shape
        self.keep_ratio = keep_ratio
        self.multiscale_mode = multiscale_mode
        self.scale_range = scale_range
        self.align_long_edge = align_long_edge
        self.interpolation = interpolation

    def __call__(self, image: NDArrayF32) -> ResizeParam:
        """Compute the parameters and put them in the data dict."""
        im_shape = (image.shape[1], image.shape[2])
        target_shape = _get_target_shape(
            im_shape,
            self.shape,
            self.keep_ratio,
            self.multiscale_mode,
            self.scale_range,
            self.align_long_edge,
        )
        scale_factor = (
            target_shape[1] / im_shape[1],
            target_shape[0] / im_shape[0],
        )
        return ResizeParam(
            target_shape=target_shape,
            scale_factor=scale_factor,
            interpolation=self.interpolation,
        )


@Transform([CK.boxes2d, "transforms.resize.scale_factor"], CK.boxes2d)
class ResizeBoxes2D:
    """Resize 2D bounding boxes."""

    def __call__(
        self, boxes: NDArrayF32, scale_factor: tuple[float, float]
    ) -> NDArrayF32:
        """Resize 2D bounding boxes.

        Args:
            boxes (Tensor): The bounding boxes to be resized.
            scale_factor (tuple[float, float]): scaling factor.

        Returns:
            Tensor: Resized bounding boxes according to parameters in resize.
        """
        boxes_ = torch.from_numpy(boxes)
        scale_matrix = torch.eye(3)
        scale_matrix[0, 0] = scale_factor[0]
        scale_matrix[1, 1] = scale_factor[1]
        return transform_bbox(scale_matrix, boxes_).numpy()


@Transform(
    [
        CK.images,
        "transforms.resize.target_shape",
        "transforms.resize.interpolation",
    ],
    CK.images,
)
class ResizeImage:
    """Resize Image."""

    def __call__(
        self,
        image: NDArrayF32,
        target_shape: tuple[int, int],
        interpolation: str = "bilinear",
    ) -> NDArrayF32:
        """Resize an image of dimensions [N, H, W, C].

        Args:
            image (Tensor): The image.
            target_shape (tuple[int, int]): The target shape after resizing.
            interpolation (str): One of nearest, bilinear, bicubic. Defaults to
                bilinear.

        Returns:
            Tensor: Resized image according to parameters in resize.
        """
        image_ = torch.from_numpy(image).permute(0, 3, 1, 2)
        image_ = _resize_tensor(
            image_, target_shape, interpolation=interpolation
        )
        return image_.permute(0, 2, 3, 1).numpy()


@Transform([CK.intrinsics, "transforms.resize.scale_factor"], CK.intrinsics)
class ResizeIntrinsics:
    """Resize Intrinsics."""

    def __call__(
        self, intrinsics: NDArrayF32, scale_factor: tuple[float, float]
    ) -> NDArrayF32:
        """Scale camera intrinsics when resizing."""
        return intrinsics[:2] * scale_factor


@Transform(
    [CK.instance_masks, "transforms.resize.target_shape"], CK.instance_masks
)
class ResizeMasks:
    """Resize instance segmentation masks."""

    def __call__(
        self, masks: NDArrayF32, target_shape: tuple[int, int]
    ) -> NDArrayF32:
        """Resize masks."""
        if len(masks) == 0:  # handle empty masks
            return masks
        masks_ = torch.from_numpy(masks)
        masks_ = (
            _resize_tensor(
                masks_.float().unsqueeze(0),
                target_shape,
                interpolation="nearest",
            )
            .type(masks_.dtype)
            .squeeze(0)
        )
        return masks_.numpy()


def _resize_tensor(
    inputs: Tensor,
    shape: tuple[int, int],
    interpolation: str = "bilinear",
) -> Tensor:
    """Resize Tensor."""
    assert interpolation in {"nearest", "bilinear", "bicubic"}
    align_corners = None if interpolation == "nearest" else False
    output = F.interpolate(
        inputs, shape, mode=interpolation, align_corners=align_corners
    )
    return output


def _get_resize_shape(
    original_shape: tuple[int, int],
    new_shape: tuple[int, int],
    keep_ratio: bool = True,
    align_long_edge: bool = False,
) -> tuple[int, int]:
    """Get shape for resize, considering keep_ratio and align_long_edge."""
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


def _get_target_shape(
    input_shape: tuple[int, int],
    shape: tuple[int, int] | list[tuple[int, int]],
    keep_ratio: bool = False,
    multiscale_mode: str = "range",
    scale_range: tuple[float, float] = (1.0, 1.0),
    align_long_edge: bool = False,
) -> tuple[int, int]:
    """Generate possibly random target shape."""
    assert multiscale_mode in {"list", "range"}
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
