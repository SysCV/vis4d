"""Pointnet++ Implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.typing import LossesType, ModelOutput
from vis4d.data.const import CommonKeys as K
from vis4d.op.base.pointnetpp import (
    PointNet2Segmentation,
    PointNet2SegmentationOut,
)


class PointNet2SegmentationModel(nn.Module):
    """PointNet++ Segmentation Model implementaiton."""

    def __init__(
        self,
        num_classes: int,
        in_dimensions: int = 3,
        weights: str | None = None,
    ):
        """Creates a Pointnet+++ Model.

        Args:
            num_classes (int): Number of classes
            in_dimensions (int, optional): Input dimensions. Defaults to 3.
            weights (str, optional): Path to weights. Defaults to None.
        """
        super().__init__()

        self.segmentation_model = PointNet2Segmentation(
            num_classes, in_dimensions
        )

        if weights is not None:
            load_model_checkpoint(self, weights)

    def forward(
        self, points3d: Tensor, semantics3d: Tensor | None = None
    ) -> PointNet2SegmentationOut | ModelOutput:
        """Forward pass of the model. Extract semantic predictions.

        Args:
            points3d (Tensor): Input point shape [b, N, C].
            semantics3d (torch.Tenosr): Groundtruth semantic labels of
                shape [b, N]. Defaults to None

        Returns:
            ModelOutput: Semantic predictions of the model.
        """
        x = self.segmentation_model(points3d)
        if semantics3d is not None:
            return x
        class_pred = torch.argmax(x.class_logits, dim=1)
        return {K.semantics3d: class_pred}


class Pointnet2SegmentationLoss(nn.Module):
    """Pointnet2SegmentationLoss Loss."""

    def __init__(
        self,
        ignore_index: int = 255,
        semantic_weights: Tensor | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            ignore_index (int, optional): Class Index that should be ignored.
                Defaults to 255.
            semantic_weights (Tensor, optional): Weights for each class.
        """
        super().__init__()
        self.segmentation_loss = nn.CrossEntropyLoss(
            weight=semantic_weights, ignore_index=ignore_index
        )

    def forward(
        self, outputs: PointNet2SegmentationOut, semantics3d: Tensor
    ) -> LossesType:
        """Calculates the loss.

        Args:
            outputs (PointNet2SegmentationOut): Model outputs.
            semantics3d (Tensor): Groundtruth semantic labels.
        """
        return dict(
            segmentation_loss=self.segmentation_loss(
                outputs.class_logits, semantics3d
            ),
        )
