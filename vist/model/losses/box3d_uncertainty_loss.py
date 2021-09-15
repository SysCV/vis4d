"""Box3d loss with uncertainty for QD-3DT."""
from typing import List

import torch
import torch.nn.functional as F
from mmdet.models.losses import smooth_l1_loss

from .base import BaseLoss, LossConfig


class Box3DUncertaintyLossConfig(LossConfig):
    """Config for Box3d loss with uncertainty."""

    loss_weights: List[float]


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
    ):
        pred = pred[torch.arange(pred.shape[0]), labels]

        # delta 2dc loss
        loss_cen = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            pred[:, :2], target[:, :2], beta=1 / 9, reduction="none"
        ).mean(dim=-1)

        # dimension loss
        loss_dim = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            pred[:, 3:6], target[:, 3:6], beta=1 / 9, reduction="none"
        ).mean(dim=-1)

        # depth loss
        depth_weights = target.new_ones(target[:, 2].shape)
        depth_weights[target[:, 2] <= 0] = 0
        loss_dep = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            pred[:, 2], target[:, 2], weight=depth_weights, reduction="none"
        )

        # rotation loss
        orientation = pred[:, 6:14]
        # bin 1
        divider1 = torch.sqrt(
            orientation[:, 2:3] ** 2 + orientation[:, 3:4] ** 2
        )
        b1sin = orientation[:, 2:3] / divider1
        b1cos = orientation[:, 3:4] / divider1

        # bin 2
        divider2 = torch.sqrt(
            orientation[:, 6:7] ** 2 + orientation[:, 7:8] ** 2
        )
        b2sin = orientation[:, 6:7] / divider2
        b2cos = orientation[:, 7:8] / divider2

        rot = torch.cat(
            [
                orientation[:, 0:2],
                b1sin,
                b1cos,
                orientation[:, 4:6],
                b2sin,
                b2cos,
            ],
            1,
        )
        loss_rot = rotation_loss(rot, target[..., 6:8], target[..., 8:]).mean(
            dim=-1
        )

        result_dict = dict(
            loss_ctr3d=self.cfg.loss_weights[0] * loss_cen,
            loss_dep3d=self.cfg.loss_weights[1] * loss_dep,
            loss_dim3d=self.cfg.loss_weights[2] * loss_dim,
            loss_rot3d=self.cfg.loss_weights[3] * loss_rot,
        )

        # uncertainty loss
        pos_depth_self_labels = torch.exp(
            -torch.abs(pred[:, 14] - target[:, 2]) * 5.0
        )
        pos_depth_self_weights = torch.where(
            pos_depth_self_labels > 0.8,
            pos_depth_self_labels.new_ones(1) * 5.0,
            pos_depth_self_labels.new_ones(1) * 0.1,
        )

        loss_unc3d = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            pred[:, 14],
            pos_depth_self_labels.detach().clone(),
            weight=pos_depth_self_weights,
            reduction="none",
            beta=1 / 9,
        ).mean(dim=-1)

        result_dict.update(
            dict(loss_unc3d=self.cfg.loss_weights[4] * loss_unc3d)
        )

        # reduce batch dimension after confidence loss computation
        for k, v in result_dict.items():
            result_dict[k] = v.mean()
        return result_dict


def rotation_loss(output, target_bin, target_res):
    """
    Rot Bin loss.

    Inputs:
        -output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
                        bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
        - target_bin: (B, 2) [bin1_cls, bin2_cls]
        - target_res: (B, 2) [bin1_res, bin2_res]
    """
    loss_bin1 = F.cross_entropy(
        output[:, 0:2], target_bin[:, 0].long(), reduction="none"
    )
    loss_bin2 = F.cross_entropy(
        output[:, 4:6], target_bin[:, 1].long(), reduction="none"
    )

    idx1 = torch.nonzero(target_bin[:, 0], as_tuple=False)[:, 0]
    idx2 = torch.nonzero(target_bin[:, 1], as_tuple=False)[:, 0]

    loss_res = torch.zeros_like(loss_bin1)
    if idx1.shape[0] > 0:
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            valid_output1[:, 2],
            torch.sin(valid_target_res1[:, 0]),
            beta=1 / 9,
            reduction="none",
        )
        loss_cos1 = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            valid_output1[:, 3],
            torch.cos(valid_target_res1[:, 0]),
            beta=1 / 9,
            reduction="none",
        )
        loss_res[idx1] += loss_sin1 + loss_cos1
    if idx2.shape[0] > 0:
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            valid_output2[:, 6],
            torch.sin(valid_target_res2[:, 1]),
            beta=1 / 9,
            reduction="none",
        )
        loss_cos2 = smooth_l1_loss(  # pylint: disable=unexpected-keyword-arg
            valid_output2[:, 7],
            torch.cos(valid_target_res2[:, 1]),
            beta=1 / 9,
            reduction="none",
        )
        loss_res[idx2] += loss_sin2 + loss_cos2

    return loss_bin1 + loss_bin2 + loss_res
