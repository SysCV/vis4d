"""RoI Pooling module from detectron2."""
from typing import List

import torch
from detectron2.modeling.poolers import ROIPooler as D2ROIPooler
from detectron2.structures import Boxes

from openmt.structures import Boxes2D

from .base_pooler import BaseRoIPooler, RoIPoolerConfig


class D2RoIPoolerConfig(RoIPoolerConfig):
    pooling_op: str
    strides: List[int]
    sampling_ratio: int


class D2RoIPooler(BaseRoIPooler):
    """detectron2 roi pooling class"""

    cfg_type = D2RoIPoolerConfig

    def __init__(self, cfg: RoIPoolerConfig):
        """Init."""
        super().__init__()
        self.cfg = D2RoIPoolerConfig(**cfg.__dict__)

        self.roi_pooler = D2ROIPooler(
            output_size=self.cfg.resolution,
            scales=[1 / s for s in self.cfg.strides],
            sampling_ratio=self.cfg.sampling_ratio,
            pooler_type=self.cfg.pooling_op,
        )

    def pool(
        self, features: List[torch.Tensor], boxes: List[Boxes2D]
    ) -> torch.Tensor:
        """detectron2 based roi pooling operation.
        Args:
            features: list of image feature tensors (e.g.
            fpn levels) - NCHW format
            boxes: list of proposals (per image)
        Returns:
            torch.Tensor: NCHW format, where N = num boxes (total),
            HW is roi size, C is feature dim.
        """
        x = self.roi_pooler(features, [Boxes(b.data[:, :4]) for b in boxes])

        return x
