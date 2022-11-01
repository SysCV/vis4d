from typing import Optional, Tuple, Union

import torch
from torch import nn

from vis4d.common.typing import LossesType, ModelOutput
from vis4d.data.const import COMMON_KEYS
from vis4d.op.base.pointnet import PointNetSegmentation, PointNetSemanticsOut
from vis4d.op.loss.orthogonal_transform_loss import (
    OrthogonalTransformRegularizationLoss,
)
from vis4d.op.util import load_model_checkpoint


class PointnetSegmentationModel(nn.Module):
    """Simple Segmentation Model using Pointnet"""

    def __init__(
        self,
        num_classes: int = 11,
        in_dimensions: int = 3,
        weights: Optional[str] = None,
    ) -> None:
        """Simple Segmentation Model using Pointnet.

        Args:
            num_classes: Number of semantic classes
            in_dimensions: Input dimension
            weights: Path to weight file
        """
        super().__init__()
        self.model = PointNetSegmentation(
            n_classes=num_classes, in_dimensions=in_dimensions
        )
        if weights is not None:
            load_model_checkpoint(self, weights)

    def __call__(
        self, data: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Union[PointNetSemanticsOut, ModelOutput]:
        """Runs the semantic model.

        Args:
            data: Input Tensor Shape [N, C, n_pts]
            target: Target Classes shape [N, n_ots]
        """
        self._call_impl(data, target)

    def forward(
        self, data: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Union[PointNetSemanticsOut, ModelOutput]:
        """Runs the semantic model.

        Args:
            data: Input Tensor Shape [N, C, n_pts]
            target: Target Classes shape [N, n_ots]
        """
        if target is not None:
            return self.forward_train(data, target)
        return self.forward_test(data)

    def forward_train(
        self,
        points: torch.Tensor,
        target: torch.Tensor,
    ) -> PointNetSemanticsOut:
        """Forward training stage."""
        out = self.model(points)
        return out

    def forward_test(
        self,
        points: torch.Tensor,
    ) -> ModelOutput:
        """Forward test stage."""
        return {
            COMMON_KEYS.semantics3d: torch.argmax(
                self.model(points).class_logits, dim=1
            )
        }


class PointnetSegmentationLoss(nn.Module):
    """PointnetSegmentationLoss Loss."""

    def __init__(
        self,
        regularize_transform=True,
        ignore_index=255,
        transform_weight: float = 1e-3,
        semantic_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Init.

        Args:
            regularize_transform: If true add transforms to loss
            ignore_index: Semantic class that should be ignored
            transform_weight: Loss weight factor for transform
                              regularization loss
            semantic_weights: Classwise weights for semantic loss
        """
        super().__init__()
        self.segmentation_loss = nn.CrossEntropyLoss(
            weight=semantic_weights, ignore_index=ignore_index
        )
        self.transformation_loss = OrthogonalTransformRegularizationLoss()
        self.regularize_transform = regularize_transform
        self.transform_weight = transform_weight

    def forward(
        self, outputs: PointNetSemanticsOut, target: torch.Tensor
    ) -> LossesType:
        """Calculates the losss.

        Args:
            outputs: Pointnet output
            target: Target Labels
        """
        if not self.regularize_transform:
            dict(
                segmentation_loss=self.segmentation_loss(
                    outputs.class_logits, target
                )
            )

        return dict(
            segmentation_loss=self.segmentation_loss(
                outputs.class_logits, target
            ),
            transform_loss=self.transform_weight
            * self.transformation_loss(outputs.transformations),
        )
