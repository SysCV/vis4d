from typing import Optional, Tuple

import torch
import torch.nn as nn

from vis4d.common.typing import COMMON_KEYS, LossesType, ModelOutput
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

    def forward(
        self, xyz, target=None
    ) -> Tuple[PointNet2Segmentation, ModelOutput]:
        """Forward pass of the model. Extract semantic predictions."""
        x = self.segmentation_model(xyz)
        class_pred = torch.argmax(x.class_logits, dim=1)
        return x, {COMMON_KEYS.semantics3d: class_pred}

    def forward_test(self, xyz) -> ModelOutput:
        """Forward test"""
        return self.forward(xyz, None)[1]

    def forward_train(self, xyz, targets) -> PointNet2Segmentation:
        """Forward train"""
        return self.forward(xyz, targets)[0]


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
        self, outputs: PointNet2SegmentationOut, target: torch.Tensor
    ) -> LossesType:
        """Calculates the loss"""
        return dict(
            segmentation_loss=self.segmentation_loss(
                outputs.class_logits, target
            ),
        )
