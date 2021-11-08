"""Box3d loss with uncertainty for QD-3DT."""
from typing import Tuple

import torch
import torch.nn.functional as F

from vis4d.struct import LossesType

from .base import BaseLoss, LossConfig
from .utils import smooth_l1_loss


class Box3DUncertaintyLossConfig(LossConfig):
    """Config for Box3d loss with uncertainty."""

    loss_weights: Tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    num_rotation_bins: int = 2


class Box3DUncertaintyLoss(BaseLoss):
    """Box3d loss for QD-3DT."""

    def __init__(self, cfg: LossConfig):
        """Init."""
        super().__init__()
        self.cfg = Box3DUncertaintyLossConfig(**cfg.dict())

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
    ) -> LossesType:
        """Compute box3d loss."""
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
            pred[:, :2], target[:, :2], beta=1 / 9, reduction="mean"
        )

        # dimension loss
        loss_dim = smooth_l1_loss(
            pred[:, 3:6], target[:, 3:6], beta=1 / 9, reduction="mean"
        )

        # depth loss
        depth_weights = (target[:, 2] > 0).float()
        loss_dep = smooth_l1_loss(
            pred[:, 2], target[:, 2], weight=depth_weights, reduction="mean"
        )

        # rotation loss
        loss_rot = rotation_loss(
            pred[:, 6 : 6 + self.cfg.num_rotation_bins * 3],
            target[:, 6 : 6 + self.cfg.num_rotation_bins],
            target[:, 6 + self.cfg.num_rotation_bins :],
            self.cfg.num_rotation_bins,
        ).mean()

        result_dict = dict(
            loss_ctr3d=self.cfg.loss_weights[0] * loss_cen,
            loss_dep3d=self.cfg.loss_weights[1] * loss_dep,
            loss_dim3d=self.cfg.loss_weights[2] * loss_dim,
            loss_rot3d=self.cfg.loss_weights[3] * loss_rot,
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
            weight=pos_depth_self_weights,
            reduction="mean",
            beta=1 / 9,
        )

        result_dict.update(
            dict(loss_unc3d=self.cfg.loss_weights[4] * loss_unc3d)
        )

        # reduce batch dimension after confidence loss computation
        for k, v in result_dict.items():
            result_dict[k] = v.mean()
        return result_dict


def rotation_loss(
    output: torch.Tensor,
    target_bin: torch.Tensor,
    target_res: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """Rotation loss.

    Consists of bin-based classification loss and residual-based regression
    loss.
    """
    loss_bins = F.binary_cross_entropy_with_logits(
        output[:, :num_bins], target_bin, reduction="none"
    ).mean(dim=-1)

    loss_res = torch.zeros_like(loss_bins)
    for i in range(num_bins):
        bin_mask = target_bin[:, i] == 1
        res_idx = num_bins + 2 * i
        if bin_mask.any():
            loss_sin = smooth_l1_loss(
                output[bin_mask, res_idx],
                torch.sin(target_res[bin_mask, i]),
                beta=1 / 9,
                reduction="none",
            )
            loss_cos = smooth_l1_loss(
                output[bin_mask, res_idx + 1],
                torch.cos(target_res[bin_mask, i]),
                beta=1 / 9,
                reduction="none",
            )
            loss_res[bin_mask] += loss_sin + loss_cos

    return loss_bins + loss_res
