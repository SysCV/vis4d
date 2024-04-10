"""VeloLSTM operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vis4d.common.typing import LossesType
from vis4d.op.loss.base import Loss


class VeloLSTMLoss(Loss):
    """Loss term for VeloLSTM."""

    def __init__(self, loc_dim: int = 7, smooth_weight: float = 0.001) -> None:
        """Initialize the loss term."""
        super().__init__()
        self.loc_dim = loc_dim
        self.smooth_weight = smooth_weight

    @staticmethod
    def linear_motion_loss(outputs: Tensor) -> Tensor:
        """Linear motion loss.

        Loss: |(loc_t - loc_t-1), (loc_t-1, loc_t-2)|_1 for t = [2, s_len]
        """
        s_len = outputs.shape[1]

        loss = outputs.new_zeros(1)
        past_motion = outputs[:, 1, :] - outputs[:, 0, :]
        for idx in range(2, s_len, 1):
            curr_motion = outputs[:, idx, :] - outputs[:, idx - 1, :]
            loss += F.l1_loss(past_motion, curr_motion, reduction="mean")
            past_motion = curr_motion
        return loss / (s_len - 2)

    def forward(
        self, loc_preds: Tensor, loc_refines: Tensor, gt_traj: Tensor
    ) -> LossesType:
        """Loss term for VeloLSTM."""
        refine_loss = F.smooth_l1_loss(
            loc_refines, gt_traj[:, 1:, : self.loc_dim], reduction="mean"
        )
        pred_loss = F.smooth_l1_loss(
            loc_preds[:, :-1, :],
            gt_traj[:, 2:, : self.loc_dim],
            reduction="mean",
        )
        linear_loss = self.linear_motion_loss(loc_preds[:, :-1, :])

        return {
            "refine_loss": refine_loss,
            "pred_loss": pred_loss,
            "linear_loss": torch.mul(self.smooth_weight, linear_loss),
        }
