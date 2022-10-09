"""Orthogonal Transform_loss."""
from typing import List

import torch
import torch.nn as nn

from .base import Loss


class OrthogonalTransformRegularizationLoss(Loss):
    """Loss that punishes linear transformations that are not orthogonal."""

    def forward(self, transforms: List[torch.Tensor]) -> torch.Tensor:
        """Calculates the loss."""
        loss = 0
        for trans in transforms:
            d = trans.size()[1]

            try:
                I = self.get_buffer(f"identity_{d}")
            except AttributeError as e:
                # Create identity buffers if not yet allocated
                I = torch.eye(d, device=trans.device)
                self.register_buffer(f"identity_{d}", I)

            loss += torch.mean(
                torch.norm(
                    torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)
                )
            )
        return loss
