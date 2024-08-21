"""Resize transformation."""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from vis4d.common.imports import OPENCV_AVAILABLE
from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import transform_bbox

from .base import Transform

if OPENCV_AVAILABLE:
    import cv2
    from cv2 import (  # pylint: disable=no-member,no-name-in-module
        INTER_AREA,
        INTER_CUBIC,
        INTER_LANCZOS4,
        INTER_LINEAR,
        INTER_NEAREST,
    )
else:
    raise ImportError("Please install opencv-python to use this module.")


class ResizeParam(TypedDict):
    """Parameters for Resize."""

    target_shape: tuple[int, int]
    scale_factor: tuple[float, float]


@Transform(K.images, ["transforms.resize", K.input_hw])
class GenResizeParameters:
    """Generate the parameters for a resize operation."""

    def __init__(
        self,
        shape: tuple[int, int] | list[tuple[int, int]],
        keep_ratio: bool = False,
        multiscale_mode: str = "range",
        scale_range: tuple[float, float] = (1.0, 1.0),
        align_long_edge: bool = False,
        resize_short_edge: bool = False,
        allow_overflow: bool = False,
        fixed_scale: bool = False,
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
            resize_short_edge (bool, optional): If keep_ratio=true, this option
                scale the image according to the short edge. e.g. original
                shape=(80, 100), shape to be resized=(100, 200) will yield
                (100, 125) as new shape. Defaults to False.
            allow_overflow (bool, optional): If set to True, we scale the image
                to the smallest size such that it is no smaller than shape.
                Otherwise, we scale the image to the largest size such that it
                is no larger than shape. Defaults to False.
            fixed_scale (bool, optional): If set to True, we scale the image
                without offset. Defaults to False.
        """
        self.shape = shape
        self.keep_ratio = keep_ratio

        assert multiscale_mode in {"list", "range"}
        self.multiscale_mode = multiscale_mode

        assert (
            scale_range[0] <= scale_range[1]
        ), f"Invalid scale range: {scale_range[1]} < {scale_range[0]}"
        self.scale_range = scale_range

        self.align_long_edge = align_long_edge
        self.resize_short_edge = resize_short_edge
        self.allow_overflow = allow_overflow
        self.fixed_scale = fixed_scale

    def _get_target_shape(
        self, input_shape: tuple[int, int]
    ) -> tuple[int, int]:
        """Generate possibly random target shape."""
        if self.multiscale_mode == "range":
            assert isinstance(
                self.shape, tuple
            ), "Specify shape as tuple when using multiscale mode range."
            if self.scale_range[0] < self.scale_range[1]:  # do multi-scale
                w_scale = (
                    random.uniform(0, 1)
                    * (self.scale_range[1] - self.scale_range[0])
                    + self.scale_range[0]
                )
                h_scale = (
                    random.uniform(0, 1)
                    * (self.scale_range[1] - self.scale_range[0])
                    + self.scale_range[0]
                )
            else:
                h_scale = w_scale = 1.0

            shape = int(self.shape[0] * h_scale), int(self.shape[1] * w_scale)
        else:
            assert isinstance(
                self.shape, list
            ), "Specify shape as list when using multiscale mode list."
            shape = random.choice(self.shape)

        return get_resize_shape(
            input_shape,
            shape,
            self.keep_ratio,
            self.align_long_edge,
            self.resize_short_edge,
            self.allow_overflow,
            self.fixed_scale,
        )

    def __call__(
        self, images: list[NDArrayF32]
    ) -> tuple[list[ResizeParam], list[tuple[int, int]]]:
        """Compute the parameters and put them in the data dict."""
        image = images[0]

        im_shape = (image.shape[1], image.shape[2])
        target_shape = self._get_target_shape(im_shape)
        scale_factor = (
            target_shape[1] / im_shape[1],
            target_shape[0] / im_shape[0],
        )

        resize_params = [
            ResizeParam(target_shape=target_shape, scale_factor=scale_factor)
        ] * len(images)
        target_shapes = [target_shape] * len(images)

        return resize_params, target_shapes


def get_resize_shape(
    original_shape: tuple[int, int],
    new_shape: tuple[int, int],
    keep_ratio: bool = True,
    align_long_edge: bool = False,
    resize_short_edge: bool = False,
    allow_overflow: bool = False,
    fixed_scale: bool = False,
) -> tuple[int, int]:
    """Get shape for resize, considering keep_ratio and align_long_edge.

    Args:
        original_shape (tuple[int, int]): Original shape in [H, W].
        new_shape (tuple[int, int]): New shape in [H, W].
        keep_ratio (bool, optional): Whether to keep the aspect ratio.
            Defaults to True.
        align_long_edge (bool, optional): Whether to align the long edge of
            the original shape with the long edge of the new shape.
            Defaults to False.
        resize_short_edge (bool, optional): Whether to resize according to the
            short edge. Defaults to False.
        allow_overflow (bool, optional): Whether to allow overflow.
            Defaults to False.
        fixed_scale (bool, optional): Whether to use fixed scale.

    Returns:
        tuple[int, int]: The new shape in [H, W].
    """
    h, w = original_shape
    new_h, new_w = new_shape

    if keep_ratio:
        if allow_overflow:
            comp_fn = max
        else:
            comp_fn = min

        if align_long_edge:
            long_edge, short_edge = max(new_shape), min(new_shape)
            scale_factor = comp_fn(
                long_edge / max(h, w), short_edge / min(h, w)
            )
        elif resize_short_edge:
            short_edge = min(original_shape)
            new_short_edge = min(new_shape)
            scale_factor = new_short_edge / short_edge
        else:
            scale_factor = comp_fn(new_w / w, new_h / h)

        if fixed_scale:
            offset = 0.0
        else:
            offset = 0.5

        new_h = int(h * scale_factor + offset)
        new_w = int(w * scale_factor + offset)

    return new_h, new_w


@Transform([K.images, "transforms.resize.target_shape"], K.images)
class ResizeImages:
    """Resize Images."""

    def __init__(
        self,
        interpolation: str = "bilinear",
        antialias: bool = False,
        imresize_backend: str = "torch",
    ) -> None:
        """Creates an instance of the class.

        Args:
            interpolation (str, optional): Interpolation method. One of
                ["nearest", "bilinear", "bicubic"]. Defaults to "bilinear".
            antialias (bool): Whether to use antialiasing. Defaults to False.
            imresize_backend (str): One of torch, cv2. Defaults to torch.
        """
        self.interpolation = interpolation
        self.antialias = antialias
        self.imresize_backend = imresize_backend
        assert imresize_backend in {
            "torch",
            "cv2",
        }, f"Invalid imresize backend: {imresize_backend}"

    def __call__(
        self, images: list[NDArrayF32], target_shapes: list[tuple[int, int]]
    ) -> list[NDArrayF32]:
        """Resize an image of dimensions [N, H, W, C].

        Args:
            image (Tensor): The image.
            target_shape (tuple[int, int]): The target shape after resizing.

        Returns:
            list[NDArrayF32]: Resized images according to parameters in resize.
        """
        for i, (image, target_shape) in enumerate(zip(images, target_shapes)):
            images[i] = resize_image(
                image,
                target_shape,
                interpolation=self.interpolation,
                antialias=self.antialias,
                backend=self.imresize_backend,
            )
        return images


def resize_image(
    inputs: NDArrayF32,
    shape: tuple[int, int],
    interpolation: str = "bilinear",
    antialias: bool = False,
    backend: str = "torch",
) -> NDArrayF32:
    """Resize image."""
    if backend == "torch":
        image = torch.from_numpy(inputs).permute(0, 3, 1, 2)
        image = resize_tensor(image, shape, interpolation, antialias)
        return image.permute(0, 2, 3, 1).numpy()

    if backend == "cv2":
        cv2_interp_codes = {
            "nearest": INTER_NEAREST,
            "bilinear": INTER_LINEAR,
            "bicubic": INTER_CUBIC,
            "area": INTER_AREA,
            "lanczos": INTER_LANCZOS4,
        }
        return cv2.resize(  # pylint: disable=no-member, unsubscriptable-object
            inputs[0].astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2_interp_codes[interpolation],
        )[None, ...].astype(np.float32)

    raise ValueError(f"Invalid imresize backend: {backend}")


@Transform([K.boxes2d, "transforms.resize.scale_factor"], K.boxes2d)
class ResizeBoxes2D:
    """Resize list of 2D bounding boxes."""

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Resize 2D bounding boxes.

        Args:
            boxes_list: (list[NDArrayF32]): The bounding boxes to be resized.
            scale_factors (list[tuple[float, float]]): scaling factors.

        Returns:
            list[NDArrayF32]: Resized bounding boxes according to parameters in
                resize.
        """
        for i, (boxes, scale_factor) in enumerate(
            zip(boxes_list, scale_factors)
        ):
            boxes_ = torch.from_numpy(boxes)
            scale_matrix = torch.eye(3)
            scale_matrix[0, 0] = scale_factor[0]
            scale_matrix[1, 1] = scale_factor[1]
            boxes_list[i] = transform_bbox(scale_matrix, boxes_).numpy()
        return boxes_list


@Transform(
    [
        K.depth_maps,
        "transforms.resize.target_shape",
        "transforms.resize.scale_factor",
    ],
    K.depth_maps,
)
class ResizeDepthMaps:
    """Resize depth maps."""

    def __init__(
        self,
        interpolation: str = "nearest",
        rescale_depth_values: bool = False,
        check_scale_factors: bool = False,
    ):
        """Initialize the transform.

        Args:
            interpolation (str, optional): Interpolation method. One of
                ["nearest", "bilinear", "bicubic"]. Defaults to "nearest".
            rescale_depth_values (bool, optional): If the depth values should
                be rescaled according to the new scale factor. Defaults to
                False. This is useful if we want to keep the intrinsic
                parameters of the camera the same.
            check_scale_factors (bool, optional): If the scale factors should
                be checked to ensure they are the same. Defaults to False.
                If False, the scale factor is assumed to be the same for both
                dimensions and will just use the first scale factor.
        """
        self.interpolation = interpolation
        self.rescale_depth_values = rescale_depth_values
        self.check_scale_factors = check_scale_factors

    def __call__(
        self,
        depth_maps: list[NDArrayF32],
        target_shapes: list[tuple[int, int]],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Resize depth maps."""
        for i, (depth_map, target_shape, scale_factor) in enumerate(
            zip(depth_maps, target_shapes, scale_factors)
        ):
            depth_map_ = torch.from_numpy(depth_map)
            depth_map_ = (
                resize_tensor(
                    depth_map_.float().unsqueeze(0).unsqueeze(0),
                    target_shape,
                    interpolation=self.interpolation,
                )
                .type(depth_map_.dtype)
                .squeeze(0)
                .squeeze(0)
            )
            if self.rescale_depth_values:
                if self.check_scale_factors:
                    assert np.isclose(
                        scale_factor[0], scale_factor[1], atol=1e-4
                    ), "Depth map scale factors must be the same"
                depth_map_ /= scale_factor[0]
            depth_maps[i] = depth_map_.numpy()
        return depth_maps


@Transform(
    [
        K.optical_flows,
        "transforms.resize.target_shape",
        "transforms.resize.scale_factor",
    ],
    K.optical_flows,
)
class ResizeOpticalFlows:
    """Resize optical flows."""

    def __init__(self, normalized_flow: bool = True):
        """Create a ResizeOpticalFlows instance.

        Args:
            normalized_flow (bool): Whether the optical flow is normalized.
                Defaults to True. If false, the optical flow will be scaled
                according to the scale factor.
        """
        self.normalized_flow = normalized_flow

    def __call__(
        self,
        optical_flows: list[NDArrayF32],
        target_shapes: list[tuple[int, int]],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Resize optical flows."""
        for i, (optical_flow, target_shape, scale_factor) in enumerate(
            zip(optical_flows, target_shapes, scale_factors)
        ):
            optical_flow_ = torch.from_numpy(optical_flow).permute(2, 0, 1)
            optical_flow_ = (
                resize_tensor(
                    optical_flow_.float().unsqueeze(0),
                    target_shape,
                    interpolation="bilinear",
                )
                .type(optical_flow_.dtype)
                .squeeze(0)
                .permute(1, 2, 0)
            )
            # scale optical flows
            if not self.normalized_flow:
                optical_flow_[:, :, 0] *= scale_factor[0]
                optical_flow_[:, :, 1] *= scale_factor[1]
            optical_flows[i] = optical_flow_.numpy()
        return optical_flows


@Transform(
    [K.instance_masks, "transforms.resize.target_shape"], K.instance_masks
)
class ResizeInstanceMasks:
    """Resize instance segmentation masks."""

    def __call__(
        self,
        masks_list: list[NDArrayF32],
        target_shapes: list[tuple[int, int]],
    ) -> list[NDArrayF32]:
        """Resize masks."""
        for i, (masks, target_shape) in enumerate(
            zip(masks_list, target_shapes)
        ):
            if len(masks) == 0:  # handle empty masks
                continue
            masks_ = torch.from_numpy(masks)
            masks_ = (
                resize_tensor(
                    masks_.float().unsqueeze(1),
                    target_shape,
                    interpolation="nearest",
                )
                .type(masks_.dtype)
                .squeeze(1)
            )
            masks_list[i] = masks_.numpy()
        return masks_list


@Transform([K.seg_masks, "transforms.resize.target_shape"], K.seg_masks)
class ResizeSegMasks:
    """Resize segmentation masks."""

    def __call__(
        self,
        masks_list: list[NDArrayF32],
        target_shape_list: list[tuple[int, int]],
    ) -> list[NDArrayF32]:
        """Resize masks."""
        for i, (masks, target_shape) in enumerate(
            zip(masks_list, target_shape_list)
        ):
            masks_ = torch.from_numpy(masks)
            masks_ = (
                resize_tensor(
                    masks_.float().unsqueeze(0).unsqueeze(0),
                    target_shape,
                    interpolation="nearest",
                )
                .type(masks_.dtype)
                .squeeze(0)
                .squeeze(0)
            )
            masks_list[i] = masks_.numpy()
        return masks_list


@Transform([K.intrinsics, "transforms.resize.scale_factor"], K.intrinsics)
class ResizeIntrinsics:
    """Resize Intrinsics."""

    def __call__(
        self,
        intrinsics: list[NDArrayF32],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Scale camera intrinsics when resizing."""
        for i, scale_factor in enumerate(scale_factors):
            scale_matrix = np.eye(3, dtype=np.float32)
            scale_matrix[0, 0] *= scale_factor[0]
            scale_matrix[1, 1] *= scale_factor[1]
            intrinsics[i] = scale_matrix @ intrinsics[i]
        return intrinsics


def resize_tensor(
    inputs: Tensor,
    shape: tuple[int, int],
    interpolation: str = "bilinear",
    antialias: bool = False,
) -> Tensor:
    """Resize Tensor."""
    assert interpolation in {"nearest", "bilinear", "bicubic"}
    align_corners = None if interpolation == "nearest" else False
    output = F.interpolate(
        inputs,
        shape,
        mode=interpolation,
        align_corners=align_corners,
        antialias=antialias,
    )
    return output
