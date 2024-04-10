"""Orthogonal Transform Loss."""

from __future__ import annotations

import torch

from .base import Loss


class OrthogonalTransformRegularizationLoss(Loss):
    """Loss that punishes linear transformations that are not orthogonal.

    Calculates difference of X'*X and identity matrix using norm( X'*X - I)
    """

    def __call___(self, transforms: list[torch.Tensor]) -> torch.Tensor:
        """Calculates the loss.

        Calculates difference of X'*X and the identity matrix using
        norm(X'*X - I) for each transformation

        Args:
            transforms: (list(torch.tensor)) list with transformation matrices
                        batched ([N, 3, 3], [N, x, x], ....)

        Returns:
            torch.Tensor containing the mean loss value (mean(norm(X'*X - I)))
        """
        return self._call_impl(transforms)

    def forward(self, transforms: list[torch.Tensor]) -> torch.Tensor:
        """Calculates the loss.

        Calculates difference of X'*X and the identity matrix using
        norm(X'*X - I) for each transformation

        Args:
            transforms: (list(torch.tensor)) list with transformation matrices
                        batched ([N, 3, 3], [N, x, x], ....)

        Returns:
            torch.Tensor containing the mean loss value (mean(norm(X'*X - I)))
        """
        loss = torch.tensor(0.0)
        for trans in transforms:
            d = trans.size()[1]

            try:
                identity = self.get_buffer(f"identity_{d}")
            except AttributeError as _:
                # Create identity buffers if not yet allocated
                identity = torch.eye(d, device=trans.device)
                self.register_buffer(f"identity_{d}", identity)

            loss += torch.mean(
                torch.norm(
                    torch.bmm(trans, trans.transpose(2, 1)) - identity,
                    dim=(1, 2),
                )
            )
        return loss
