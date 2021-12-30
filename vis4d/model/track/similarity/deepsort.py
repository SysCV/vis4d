"""Similarity Head for quasi-dense instance similarity learning."""
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from vis4d.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from vis4d.common.bbox.samplers import SamplingResult
from vis4d.common.layers import BasicBlock, Conv2d
from vis4d.model.losses import LossConfig, build_loss
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
)

from .base import BaseSimilarityHead, SimilarityLearningConfig


class DeepSortSimilarityHeadConfig(SimilarityLearningConfig):
    """Deep Sort Similarity Head config."""

    num_instances: int = 625
    num_fcs: int = 1
    proj_dim: int = 32
    fc_out_dim: int = 128
    max_boxes_num: int = 512
    drop_prob: float = 0.6
    loss_cls: LossConfig
    roi_align_config: RoIPoolerConfig
    backbone: Optional[str]


def make_layers(
    c_in: int, c_out: int, repeat_times: int = 2, is_downsample: bool = False
) -> torch.nn.modules.container.Sequential:
    """Make layers."""
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [
                BasicBlock(c_in, c_out, is_downsample=is_downsample),
            ]
        else:
            blocks += [
                BasicBlock(c_out, c_out),
            ]
    return nn.Sequential(*blocks)


class DeepSortSimilarityHead(BaseSimilarityHead):
    """DeepSort embedding head for ReID similarity learning."""

    def __init__(self, cfg: SimilarityLearningConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = DeepSortSimilarityHeadConfig(**cfg.dict())
        self.roi_pooler = build_roi_pooler(self.cfg.roi_align_config)
        self.fcs = self._init_layers()

        self.backbone = nn.Sequential(
            Conv2d(
                3,
                self.cfg.proj_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                norm=getattr(nn, "BatchNorm2d")(self.cfg.proj_dim),
                activation=nn.ELU(inplace=True),
            ),
            Conv2d(
                self.cfg.proj_dim,
                self.cfg.proj_dim,
                kernel_size=3,
                padding=1,
                stride=1,
                norm=getattr(nn, "BatchNorm2d")(self.cfg.proj_dim),
                activation=nn.ELU(inplace=True),
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            make_layers(
                self.cfg.proj_dim, self.cfg.proj_dim, is_downsample=False
            ),
            make_layers(
                self.cfg.proj_dim,
                self.cfg.proj_dim * 2,
                is_downsample=True,
            ),
            make_layers(
                self.cfg.proj_dim * 2,
                self.cfg.proj_dim * 4,
                is_downsample=True,
            ),
        )
        self.classifier = nn.Linear(
            self.cfg.fc_out_dim, self.cfg.num_instances
        )
        self.loss_cls = build_loss(self.cfg.loss_cls)

    def _init_layers(
        self,
    ) -> torch.nn.ModuleList:
        """Init modules of head."""
        fcs = nn.ModuleList()
        if self.cfg.num_fcs > 0:
            input_dim = (
                self.cfg.proj_dim
                * 4
                * np.prod(self.cfg.roi_align_config.resolution)
                // 64
            )
            for _ in range(self.cfg.num_fcs):
                fcs.append(
                    nn.Sequential(
                        nn.Dropout(p=self.cfg.drop_prob),
                        nn.Linear(input_dim, self.cfg.fc_out_dim),
                        nn.BatchNorm1d(self.cfg.fc_out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
        return fcs

    def _head_forward(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Similarity head forward pass.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Detected boxes to apply similarity learning on.
            indices: indices order for images and labels.

        Returns:
            torch.Tensor: embedding after feature backbone extractor.
        """
        x = self.roi_pooler.pool([inputs], boxes)
        if indices is not None:
            x = x[indices]

        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)

        return x

    def forward_train(
        self,
        inputs: List[InputSample],
        boxes: List[List[Boxes2D]],
        features: Optional[List[FeatureMaps]],
        targets: List[LabelInstances],
    ) -> Tuple[LossesType, Optional[List[SamplingResult]]]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched, including
                possible reference views. The keyframe is at index 0.
            boxes: Detected boxes to apply similarity learning on.
            features: Input feature maps. Batched, including possible
                reference views. The keyframe is at index 0.
            targets: Corresponding targets to each InputSample.

        Returns:
            LossesType: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling results.
        """
        instance_ids = torch.cat(
            [label.track_ids for label in boxes[0]], dim=0
        ).to(dtype=torch.long)
        batch_size = min(self.cfg.max_boxes_num, len(instance_ids))
        indices = torch.randperm(len(instance_ids))[:batch_size]
        instance_ids = instance_ids[indices]

        x = self._head_forward(inputs[0], boxes[0], indices)

        if self.cfg.num_fcs > 0:
            for fc in self.fcs:
                x = fc(x)

        cls_score = self.classifier(x)
        return {"ce_loss": self.loss_cls(cls_score, instance_ids)}, None

    def forward_test(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: Optional[FeatureMaps],
    ) -> List[torch.Tensor]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Input boxes to compute similarity embedding for.
            features: Input feature maps. Batched.

        Returns:
            List[torch.Tensor]: Similarity embeddings (one vector per box, one
            tensor per batch element).
        """
        x = self._head_forward(inputs, boxes)
        if self.cfg.num_fcs > 0:
            for fc in self.fcs:
                x = fc[1](x)
        return [x.div(x.norm(p=2, dim=1, keepdim=True))]
