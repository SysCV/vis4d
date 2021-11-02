"""Vis4D RoI Pooling module."""
import math
from typing import List

import torch
from torchvision.ops import roi_align, roi_pool

from vis4d.struct import Boxes2D

from .base import BaseRoIPooler, RoIPoolerConfig
from .utils import assign_boxes_to_levels, boxes_to_tensor


class MultiScaleRoIPoolerConfig(RoIPoolerConfig):
    """MultiScaleRoIPoolerConfig config.

    strides: feature map strides relative to the input.
        The strides must be powers of 2 and a monotically decreasing geometric
         sequence with a factor of 1/2.
    sampling_ratio: Parameter for the RoIAlign (see torchvision).
    pooler_type: Name of pooling operation that should be applied. "RoIPool" or
        "RoIAlign".
    canonical_box_size: A canonical box size in pixels (sqrt(box area)). The
        default is heuristically defined as 224 pixels in the FPN paper
        (based on ImageNet pre-training).
    canonical_level (int): The feature map level index from which a canonical-
        sized box should be placed. The default is defined as level 4
        (stride=16) in the FPN paper, i.e., a box of size 224x224 will be
        placed on the feature with stride=16.
        The box placement for all boxes will be determined from their sizes
        w.r.t canonical_box_size. For example, a box whose area is 4x that of a
        canonical box should be used to pool features from feature level
        ``canonical_level+1``.
    aligned (bool): For roi_align op. Shift the box coordinates it by -0.5
        for a better alignment with the two neighboring pixel indices.
    """

    pooling_op: str
    strides: List[int]
    sampling_ratio: int
    canonical_box_size: int = 224
    canonical_level: int = 4
    aligned: bool = True


# implementation modified from:
# https://github.com/facebookresearch/detectron2/
class MultiScaleRoIPooler(BaseRoIPooler):
    """Vis4D wrapper for torchvision roi pooling class.

    Supports multiple scales.
    """

    def __init__(self, cfg: MultiScaleRoIPoolerConfig):
        """Init."""
        super().__init__()
        self.cfg = MultiScaleRoIPoolerConfig(**cfg.dict())

        assert self.cfg.pooling_op in [
            "RoIAlign",
            "RoIPool",
        ], f"Unknown pooling_op: {self.cfg.pooling_op}"

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        self.scales = [1 / s for s in self.cfg.strides]

        min_level = -(math.log2(self.scales[0]))
        max_level = -(math.log2(self.scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(self.scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert self.min_level >= 0 and self.min_level <= self.max_level
        assert self.cfg.canonical_box_size > 0

    def pool(
        self, features: List[torch.Tensor], boxes: List[Boxes2D]
    ) -> torch.Tensor:
        """Torchvision based roi pooling operation.

        Args:
            features: list of image feature tensors (e.g.
            fpn levels) - NCHW format
            boxes: list of proposals (per image)

        Returns:
            torch.Tensor: NCHW format, where N = num boxes (total),
            HW is roi size, C is feature dim. Boxes are concatenated along
            dimension 0 for all batch elements.
        """
        assert len(features) == len(self.scales), (
            f"unequal value, len(strides)={len(self.scales)}, "
            f"but x is list of {len(features)} Tensors"
        )

        assert len(boxes) == features[0].shape[0], (
            f"unequal value, x[0] batch dim 0 is {features[0].shape[0]}, "
            f"but box_list has length {len(boxes)}"
        )
        if len(boxes) == 0:
            return torch.zeros(
                (0, features[0].shape[1]) + self.cfg.resolution,
                device=features[0].device,
                dtype=features[0].dtype,
            )

        pooler_fmt_boxes = boxes_to_tensor(boxes)
        if len(self.cfg.strides) == 1:
            return self._pooling_op(
                features[0],
                pooler_fmt_boxes,
                spatial_scale=self.scales[0],
            )

        level_assignments = assign_boxes_to_levels(
            boxes,
            self.min_level,
            self.max_level,
            self.cfg.canonical_box_size,
            self.cfg.canonical_level,
        )

        num_boxes = pooler_fmt_boxes.shape[0]
        num_channels = features[0].shape[1]
        output_size = self.cfg.resolution[0]

        dtype, device = features[0].dtype, features[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )

        for level, scale in enumerate(self.scales):
            inds = (level_assignments == level).nonzero()[:, 0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            pooled_features = self._pooling_op(
                features[level],
                pooler_fmt_boxes_level,
                spatial_scale=scale,
            )
            # Use index_put_ instead of advance indexing
            # avoids pytorch/issues/49852
            output.index_put_((inds,), pooled_features)

        return output

    def _pooling_op(
        self,
        inputs: torch.Tensor,
        boxes: torch.Tensor,
        spatial_scale: float = 1.0,
    ) -> torch.Tensor:
        """Execute pooling op defined in config."""
        if self.cfg.pooling_op == "RoIAlign":
            return roi_align(
                inputs,
                boxes,
                self.cfg.resolution,
                spatial_scale,
                self.cfg.sampling_ratio,
                self.cfg.aligned,
            )
        if self.cfg.pooling_op == "RoIPool":
            return roi_pool(inputs, boxes, self.cfg.resolution, spatial_scale)
        raise ValueError(f"Unknown pooling_op: {self.cfg.pooling_op}")
