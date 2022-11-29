"""Box3d loss with uncertainty for QD-3DT."""
from __future__ import annotations

import torch

from vis4d.common import LossesType

from .base import Loss
from .common import rotation_loss, smooth_l1_loss
from .reducer import LossReducer, SumWeightedLoss, mean_loss


class Box3DUncertaintyLoss(Loss):
    """Box3d loss for QD-3DT."""

    def __init__(
        self,
        reducer: LossReducer = mean_loss,
        loss_weights: tuple[float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        num_rotation_bins: int = 2,
    ) -> None:
        """Init.

        Args:
            reducer (LossReducer): Reducer for the loss function.
            loss_weights (tuple[float x 5]): Weights for each loss term in
                the order 'delta 2dc', 'dimension', 'depth', 'rotation' and
                'uncertainty'
            num_rotation_bins (int): Number of rotation bins.
        """
        super().__init__(reducer)
        self.loss_weights = loss_weights
        self.num_rotation_bins = num_rotation_bins

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, labels: torch.Tensor
    ) -> LossesType:
        """Compute box3d loss.

        Args:
            pred (torch.Tensor): Box predictions of shape
                [N, 6 + 3 * num_rotations_bins].
            target (torcch.Tensor): Target boxes of shape
                [N, 6 + num_rotation_bins].
            labels (torch.Tensor): Target Labels of shape [N, 1].

        Returns:
           dict[str, Tensor] containing 'delta 2dc', 'dimension', 'depth',
             'rotation' and 'uncertainty' loss.
        """
        if pred.size(0) == 0:
            loss_ctr3d = loss_dep3d = loss_dim3d = loss_rot3d = loss_conf3d = (
                pred.sum() * 0
            )
            result_dict = dict(
                loss_ctr3d=loss_ctr3d,
                loss_dep3d=loss_dep3d,
                loss_dim3d=loss_dim3d,
                loss_rot3d=loss_rot3d,
                loss_conf3d=loss_conf3d,
            )

            return result_dict

        pred = pred[torch.arange(pred.shape[0], device=pred.device), labels]

        # delta 2dc loss
        loss_cen = smooth_l1_loss(
            pred[:, :2],
            target[:, :2],
            reducer=self.reducer,
            beta=1 / 9,
        )

        # dimension loss
        dim_mask = target[:, 3:6] != 100.0
        loss_dim = smooth_l1_loss(
            pred[:, 3:6][dim_mask],
            target[:, 3:6][dim_mask],
            reducer=self.reducer,
            beta=1 / 9,
        )

        # depth loss
        depth_mask = target[:, 2] > 0
        print(depth_mask)
        loss_dep = smooth_l1_loss(
            pred[:, 2][depth_mask],
            target[:, 2][depth_mask],
            reducer=self.reducer,
            beta=1 / 9,
        )

        # rotation loss
        loss_rot = rotation_loss(
            pred[:, 6 : 6 + self.num_rotation_bins * 3],
            target[:, 6 : 6 + self.num_rotation_bins],
            target[:, 6 + self.num_rotation_bins :],
            self.num_rotation_bins,
        )
        # uncertainty loss
        pos_depth_self_labels = torch.exp(
            -torch.abs(pred[:, 2] - target[:, 2]) * 5.0
        )
        pos_depth_self_weights = torch.where(
            pos_depth_self_labels > 0.8,
            pos_depth_self_labels.new_ones(1) * 5.0,
            pos_depth_self_labels.new_ones(1) * 0.1,
        )

        loss_unc3d = smooth_l1_loss(
            pred[:, -1],
            pos_depth_self_labels.detach().clone(),
            reducer=SumWeightedLoss(
                pos_depth_self_weights, 1 / len(pos_depth_self_weights)
            ),
            beta=1 / 9,
        )

        return dict(
            loss_ctr3d=self.loss_weights[0] * loss_cen,
            loss_dep3d=self.loss_weights[1] * loss_dep,
            loss_dim3d=self.loss_weights[2] * loss_dim,
            loss_rot3d=self.loss_weights[3] * loss_rot,
            loss_unc3d=self.loss_weights[4] * loss_unc3d,
        )
