"""Pointnet++ Implementation."""
from typing import Optional, Union, overload

import torch
import torch.nn as nn

from vis4d.common.typing import LossesType, ModelOutput
from vis4d.data.const import COMMON_KEYS
from vis4d.op.base.pointnetpp import (
    PointNet2Segmentation,
    PointNet2SegmentationOut,
)
from vis4d.op.util import load_model_checkpoint


class PointNet2SegmentationModel(nn.Module):
    """PointNet++ Segmentation Model implementaiton."""

    def __init__(
        self,
        num_classes,
        in_dimensions: int = 3,
        weights: Optional[str] = None,
    ):
        super().__init__()

        self.segmentation_model = PointNet2Segmentation(
            num_classes, in_dimensions
        )

        if weights is not None:
            load_model_checkpoint(self, weights)

    @overload
    def forward(self, points3d: torch.Tensor) -> ModelOutput:
        ...

    @overload
    def forward(
        self, points3d: torch.Tensor, semantics3d: Optional[torch.Tensor]
    ) -> PointNet2SegmentationOut:
        ...

    def forward(
        self, points3d: torch.Tensor, semantics3d=None
    ) -> Union[PointNet2SegmentationOut, ModelOutput]:
        """Forward pass of the model. Extract semantic predictions."""
        x = self.segmentation_model(points3d)
        if semantics3d is not None:
            return x
        class_pred = torch.argmax(x.class_logits, dim=1)
        return {COMMON_KEYS.semantics3d: class_pred}

    def forward_test(self, points3d) -> ModelOutput:
        """Forward test."""
        return self.forward(points3d)

    def forward_train(self, points3d, semantics3d) -> PointNet2SegmentationOut:
        """Forward train."""
        return self.forward(points3d, semantics3d)


class Pointnet2SegmentationLoss(nn.Module):
    """Pointnet2SegmentationLoss Loss."""

    def __init__(
        self, ignore_index=255, semantic_weights: Optional[torch.Tensor] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.segmentation_loss = nn.CrossEntropyLoss(
            weight=semantic_weights, ignore_index=ignore_index
        )

    def forward(
        self, outputs: PointNet2SegmentationOut, semantics3d: torch.Tensor
    ) -> LossesType:
        """Calculates the loss"""
        return dict(
            segmentation_loss=self.segmentation_loss(
                outputs.class_logits, semantics3d
            ),
        )
