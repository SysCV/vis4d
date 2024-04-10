"""Point generator for 2D bounding boxes.

Modified from:
https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/point_generator.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.modules.utils import _pair

from .util import meshgrid


class MlvlPointGenerator:
    """Standard points generator for multi-level feature maps.

    Used for 2D points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(
        self, strides: list[int] | list[tuple[int, int]], offset: float = 0.5
    ):
        """Init."""
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        """Number of feature levels."""
        return len(self.strides)

    @property
    def num_base_priors(self) -> list[int]:
        """Number of points at a point on the feature grid."""
        return [1 for _ in range(len(self.strides))]

    def grid_priors(
        self,
        featmap_sizes: list[tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
        with_stride: bool = False,
    ) -> list[torch.Tensor]:
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple[int, int]]): List of feature map sizes in
                multiple feature levels, each (H, W).
            dtype (torch.dtype): Dtype of priors. Defaults to torch.float32.
            device (torch.device): The device where the anchors will be put on.
                Defaults to torch.device("cuda").
            with_stride (bool): Whether to concatenate the stride to the last
                dimension of points. Defaults to False,

        Return:
            list[torch.Tensor]: Points of multiple feature levels.
                The sizes of each tensor should be (N, 2) when with stride is
                ``False``, where N = width * height, width and height
                are the sizes of the corresponding feature level,
                and the last dimension 2 represent (coord_x, coord_y),
                otherwise the shape should be (N, 4),
                and the last dimension 4 represent
                (coord_x, coord_y, stride_w, stride_h).
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride,
            )
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(
        self,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
        with_stride: bool = False,
    ) -> torch.Tensor:
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int, int]): Size of the feature maps, (H, W).
            level_idx (int): The index of corresponding feature map level.
            dtype (torch.dtype): Dtype of priors. Defaults to torch.float32.
            device (torch.device): The device where the tensors will be put on.
                Defaults to torch.device("cuda").
            with_stride (bool): Concatenate the stride to the last dimension
                of points. Defaults to False,

        Return:
            Tensor: Points of single feature levels.
                The shape of tensor should be (N, 2) when with stride is
                ``False``, where N = width * height, width and height
                are the sizes of the corresponding feature level,
                and the last dimension 2 represent (coord_x, coord_y),
                otherwise the shape should be (N, 4),
                and the last dimension 4 represent
                (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (
            torch.arange(0, feat_w, device=device) + self.offset
        ) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (
            torch.arange(0, feat_h, device=device) + self.offset
        ) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(
                dtype
            )
            stride_h = shift_xx.new_full((shift_yy.shape[0],), stride_h).to(
                dtype
            )
            shifts = torch.stack(
                [shift_xx, shift_yy, stride_w, stride_h], dim=-1
            )
        all_points = shifts.to(device)
        return all_points

    def valid_flags(
        self,
        featmap_sizes: list[tuple[int, int]],
        pad_shape: tuple[int, int],
        device: torch.device = torch.device("cuda"),
    ) -> list[torch.Tensor]:
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple[int, int]]): List of feature map sizes in
                multiple feature levels, each (H, W).
            pad_shape (tuple[int, int]): The padded shape of the image, (H, W).
            device (torch.device): The device where the anchors will be put on.
                Defaults to torch.device("cuda").

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w), device=device
            )
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(
        self,
        featmap_size: tuple[int, int],
        valid_size: tuple[int, int],
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int, int]): The size of feature maps, (H, W).
            valid_size (tuple[int, int]): The valid size of the feature maps,
                (H, W).
            device (torch.device, optional): The device where the flags will
                be put on. Defaults to torch.device("cuda").

        Returns:
            torch.Tensor: The valid flags of each points in a single level
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid
