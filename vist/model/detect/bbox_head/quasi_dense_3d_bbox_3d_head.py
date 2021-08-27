"""Similarity Head definition for quasi-dense instance similarity learning."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.layers.batch_norm import get_norm
from detectron2.layers.wrappers import Conv2d
from mmdet.models import build_loss as build_mmdet_loss

from torch import nn

from vist.common.bbox.utils import multi_apply
from vist.model.detect.losses import LossConfig, build_loss

from .base import BaseBoundingBoxConfig, BaseBoundingBoxHead
import pdb


class QD3DBBox3DHeadConfig(BaseBoundingBoxConfig):
    """QD-3DT 3D Bounding Box Head config."""

    num_shared_convs: int
    num_shared_fcs: int
    num_dep_convs: int
    num_dep_fcs: int
    num_dim_convs: int
    num_dim_fcs: int
    num_rot_convs: int
    num_rot_fcs: int
    num_2dc_convs: int
    num_2dc_fcs: int
    in_channels: int
    conv_out_dim: int
    fc_out_dim: int
    roi_feat_size: int
    num_classes: int
    target_means: List[float]
    target_stds: List[float]
    center_scale: float
    reg_class_agnostic: bool
    conv_has_bias: bool
    norm: str
    with_depth: bool
    with_uncertainty: bool
    with_dim: bool
    with_rot: bool
    with_2dc: bool

    loss_depth: dict
    loss_dim: dict
    loss_uncertainty: dict
    loss_rot: LossConfig
    loss_2dc: dict


class QD3DBBox3DHead(BaseBoundingBoxHead):
    """QD-3DT 3D Bounding Box Head."""

    def __init__(self, cfg: BaseBoundingBoxConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QD3DBBox3DHeadConfig(**cfg.dict())

        self.cls_out_channels = self.cfg.num_classes

        # add shared convs and fcs
        (
            self.shared_convs,
            self.shared_fcs,
            self.shared_out_channels,
        ) = self._add_conv_fc_branch(
            self.cfg.num_shared_convs,
            self.cfg.num_shared_fcs,
            self.cfg.in_channels,
            True,
        )

        # add depth specific branch
        (
            self.dep_convs,
            self.dep_fcs,
            self.dep_last_dim,
        ) = self._add_conv_fc_branch(
            self.cfg.num_dep_convs,
            self.cfg.num_dep_fcs,
            self.shared_out_channels,
        )

        # add dim specific branch
        (
            self.dim_convs,
            self.dim_fcs,
            self.dim_last_dim,
        ) = self._add_conv_fc_branch(
            self.cfg.num_dim_convs,
            self.cfg.num_dim_fcs,
            self.shared_out_channels,
        )

        # add rot specific branch
        (
            self.rot_convs,
            self.rot_fcs,
            self.rot_last_dim,
        ) = self._add_conv_fc_branch(
            self.cfg.num_rot_convs,
            self.cfg.num_rot_fcs,
            self.shared_out_channels,
        )

        # add 2dc specific branch
        (
            self.cen_2d_convs,
            self.cen_2d_fcs,
            self.cen_2d_last_dim,
        ) = self._add_conv_fc_branch(
            self.cfg.num_2dc_convs,
            self.cfg.num_2dc_fcs,
            self.shared_out_channels,
        )

        if self.cfg.num_shared_fcs == 0:
            if self.cfg.num_dep_fcs == 0:
                self.dep_last_dim *= (
                    self.cfg.roi_feat_size * self.cfg.roi_feat_size
                )
            if self.cfg.num_dim_fcs == 0:
                self.dim_last_dim *= (
                    self.cfg.roi_feat_size * self.cfg.roi_feat_size
                )
            if self.cfg.num_rot_fcs == 0:
                self.rot_last_dim *= (
                    self.cfg.roi_feat_size * self.cfg.roi_feat_size
                )
            if self.cfg.num_2dc_fcs == 0:
                self.cen_2d_last_dim *= (
                    self.cfg.roi_feat_size * self.cfg.roi_feat_size
                )

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.cfg.with_depth:
            out_dim_dep = (
                1 if self.cfg.reg_class_agnostic else self.cls_out_channels
            )
            if self.cfg.with_uncertainty:
                self.fc_dep_uncer = nn.Linear(self.dep_last_dim, out_dim_dep)
            self.fc_dep = nn.Linear(self.dep_last_dim, out_dim_dep)
        if self.cfg.with_dim:
            out_dim_size = (
                3 if self.cfg.reg_class_agnostic else 3 * self.cls_out_channels
            )
            self.fc_dim = nn.Linear(self.dim_last_dim, out_dim_size)
        if self.cfg.with_rot:
            out_rot_size = (
                8 if self.cfg.reg_class_agnostic else 8 * self.cls_out_channels
            )
            self.fc_rot = nn.Linear(self.rot_last_dim, out_rot_size)
        if self.cfg.with_2dc:
            out_2dc_size = (
                2 if self.cfg.reg_class_agnostic else 2 * self.cls_out_channels
            )
            self.fc_2dc = nn.Linear(self.cen_2d_last_dim, out_2dc_size)

        self._init_weights()

        # losses
        # TODO: Reorganize detection and tracking loss
        self.loss_depth = (
            build_mmdet_loss(self.cfg.loss_depth)
            if self.cfg.with_depth
            else None
        )
        self.loss_dim = (
            build_mmdet_loss(self.cfg.loss_dim) if self.cfg.with_dim else None
        )
        self.loss_rot = (
            build_loss(self.cfg.loss_rot) if self.cfg.with_rot else None
        )
        self.loss_2dc = (
            build_mmdet_loss(self.cfg.loss_2dc) if self.cfg.with_2dc else None
        )
        self.loss_uncertainty = (
            build_mmdet_loss(self.cfg.loss_uncertainty)
            if self.cfg.with_uncertainty
            else None
        )

    def _init_weights(self) -> None:
        """Init weights of modules in head."""
        module_lists = [self.shared_fcs]
        if self.cfg.with_depth:
            if self.cfg.with_uncertainty:
                module_lists += [self.fc_dep_uncer]
            module_lists += [self.fc_dep, self.dep_fcs]
        if self.cfg.with_dim:
            module_lists += [self.fc_dim, self.dim_fcs]
        if self.cfg.with_rot:
            module_lists += [self.fc_rot, self.rot_fcs]
        if self.cfg.with_2dc:
            module_lists += [self.fc_2dc, self.cen_2d_fcs]

        for module_list in module_lists:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def _add_conv_fc_branch(
        self,
        num_branch_convs: int,
        num_branch_fcs: int,
        in_channels: int,
        is_shared: bool = False,
    ) -> Tuple[torch.nn.ModuleList, torch.nn.ModuleList, int]:
        """Init modules of head."""
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_dim = (
                    last_layer_dim if i == 0 else self.cfg.conv_out_dim
                )
                convs.append(
                    Conv2d(
                        conv_in_dim,
                        self.cfg.conv_out_dim,
                        kernel_size=3,
                        padding=1,
                        bias=self.cfg.conv_has_bias,
                        norm=get_norm(self.cfg.norm, self.cfg.conv_out_dim),
                        activation=nn.ReLU(inplace=True),
                    )
                )
            last_layer_dim = self.cfg.conv_out_dim

        fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            last_layer_dim *= np.prod(self.cfg.proposal_pooler.resolution)
            for i in range(num_branch_fcs):
                fc_in_dim = last_layer_dim if i == 0 else self.cfg.fc_out_dim
                fcs.append(
                    nn.Sequential(
                        nn.Linear(fc_in_dim, self.cfg.fc_out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
            last_layer_dim = self.cfg.fc_out_dim
        return convs, fcs, last_layer_dim

    def get_embeds(self, x):
        """Generate embedding from bbox feature."""
        # shared part
        if self.cfg.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.cfg.num_shared_fcs > 0:
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_dep = x
        x_dim = x
        x_rot = x
        x_2dc = x

        for conv in self.dep_convs:
            x_dep = conv(x_dep)
        if x_dep.dim() > 2:
            x_dep = x_dep.view(x_dep.size(0), -1)
        for fc in self.dep_fcs:
            x_dep = self.relu(fc(x_dep))

        for conv in self.dim_convs:
            x_dim = conv(x_dim)
        if x_dim.dim() > 2:
            x_dim = x_dim.view(x_dim.size(0), -1)
        for fc in self.dim_fcs:
            x_dim = self.relu(fc(x_dim))

        for conv in self.rot_convs:
            x_rot = conv(x_rot)
        if x_rot.dim() > 2:
            x_rot = x_rot.view(x_rot.size(0), -1)
        for fc in self.rot_fcs:
            x_rot = self.relu(fc(x_rot))

        for conv in self.cen_2d_convs:
            x_2dc = conv(x_2dc)
        if x_2dc.dim() > 2:
            x_2dc = x_2dc.view(x_2dc.size(0), -1)
        for fc in self.cen_2d_fcs:
            x_2dc = self.relu(fc(x_2dc))

        return x_dep, x_dim, x_rot, x_2dc

    def get_logits(self, x_dep, x_dim, x_rot, x_2dc):
        """Generate logits for bbox 3d."""

        def get_rot(pred):
            """Estimate alpha from bins."""
            pred = pred.view(pred.size(0), -1, 8)

            # bin 1
            divider1 = torch.sqrt(
                pred[:, :, 2:3] ** 2 + pred[:, :, 3:4] ** 2 + 1e-10
            )
            b1sin = pred[:, :, 2:3] / divider1
            b1cos = pred[:, :, 3:4] / divider1

            # bin 2
            divider2 = torch.sqrt(
                pred[:, :, 6:7] ** 2 + pred[:, :, 7:8] ** 2 + 1e-10
            )
            b2sin = pred[:, :, 6:7] / divider2
            b2cos = pred[:, :, 7:8] / divider2

            rot = torch.cat(
                [pred[:, :, 0:2], b1sin, b1cos, pred[:, :, 4:6], b2sin, b2cos],
                2,
            )
            return rot

        depth_preds = self.fc_dep(x_dep) if self.cfg.with_depth else None
        depth_uncertainty_preds = (
            self.fc_dep_uncer(x_dep)
            if self.cfg.with_depth and self.cfg.with_uncertainty
            else None
        )
        dim_preds = self.fc_dim(x_dim) if self.cfg.with_dim else None
        alpha_preds = (
            get_rot(self.fc_rot(x_rot)) if self.cfg.with_rot else None
        )
        delta_2dc_preds = self.fc_2dc(x_2dc) if self.cfg.with_2dc else None

        return (
            depth_preds,
            depth_uncertainty_preds,
            dim_preds,
            alpha_preds,
            delta_2dc_preds,
        )

    def forward(self, x):
        """Forward bbox 3d estimation."""
        x_dep, x_dim, x_rot, x_2dc = self.get_embeds(x)
        (
            depth_preds,
            depth_uncertainty_preds,
            dim_preds,
            alpha_preds,
            delta_2dc_preds,
        ) = self.get_logits(x_dep, x_dim, x_rot, x_2dc)

        return (
            depth_preds,
            depth_uncertainty_preds,
            dim_preds,
            alpha_preds,
            delta_2dc_preds,
        )

    def get_target(
        self,
        sampling_results,
        targets_3d,
        concat: bool = True,
    ):
        """Prepare 3D target for training."""
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_depths_list = [
            t.depths[res.pos_assigned_gt_inds]
            for t, res in zip(targets_3d, sampling_results)
        ]
        pos_gt_alphas_list = [
            t.alphas[res.pos_assigned_gt_inds]
            for t, res in zip(targets_3d, sampling_results)
        ]
        pos_gt_dims_list = [
            t.dims[res.pos_assigned_gt_inds]
            for t, res in zip(targets_3d, sampling_results)
        ]
        pos_gt_delta_2dcs_list = [
            t.delta_2dcs[res.pos_assigned_gt_inds]
            for t, res in zip(targets_3d, sampling_results)
        ]

        (
            labels,
            depth_targets,
            depth_weights,
            roty_targets,
            roty_weights,
            dim_targets,
            dim_weights,
            delta_2dc_targets,
            delta_2dc_weights,
        ) = multi_apply(
            self.get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_labels_list,
            pos_gt_depths_list,
            pos_gt_alphas_list,
            pos_gt_dims_list,
            pos_gt_delta_2dcs_list,
        )

        if concat:
            labels = torch.cat(labels, 0)
            depth_targets = torch.cat(depth_targets, 0)
            depth_weights = torch.cat(depth_weights, 0)
            roty_targets = torch.cat(roty_targets, 0)
            roty_weights = torch.cat(roty_weights, 0)
            dim_targets = torch.cat(dim_targets, 0)
            dim_weights = torch.cat(dim_weights, 0)
            delta_2dc_targets = torch.cat(delta_2dc_targets, 0)
            delta_2dc_weights = torch.cat(delta_2dc_weights, 0)

        return (
            labels,
            depth_targets,
            depth_weights,
            roty_targets,
            roty_weights,
            dim_targets,
            dim_weights,
            delta_2dc_targets,
            delta_2dc_weights,
        )

    def get_target_single(
        self,
        pos_bboxes,
        neg_bboxes,
        pos_gt_labels,
        pos_gt_depths,
        pos_gt_alphas,
        pos_gt_dims,
        pos_gt_delta_2dcs,
    ):
        """Prepare single 3D target for training."""
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
        depth_targets = pos_bboxes.new_zeros(num_samples, 1)
        depth_weights = pos_bboxes.new_zeros(num_samples, 1)
        roty_targets = pos_bboxes.new_zeros(num_samples, 1)
        roty_weights = pos_bboxes.new_zeros(num_samples, 1)
        dim_targets = pos_bboxes.new_zeros(num_samples, 3)
        dim_weights = pos_bboxes.new_zeros(num_samples, 3)
        delta_2dc_targets = pos_bboxes.new_zeros(num_samples, 2)
        delta_2dc_weights = pos_bboxes.new_zeros(num_samples, 2)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            if pos_gt_depths is not None:
                depth_targets[:num_pos] = pos_gt_depths
                depth_weights[:num_pos] = 1.0

            if pos_gt_alphas is not None:
                roty_targets[:num_pos] = pos_gt_alphas[:num_pos]
                roty_weights[:num_pos] = 1.0

            if pos_gt_dims is not None:
                dim_targets[:num_pos, :] = pos_gt_dims[:num_pos]
                dim_weights[:num_pos, :] = 1.0

            if pos_gt_delta_2dcs is not None:
                delta_2dc_targets[:num_pos, :] = pos_gt_delta_2dcs[:num_pos]
                delta_2dc_weights[:num_pos, :] = 1.0

        return (
            labels,
            depth_targets,
            depth_weights,
            roty_targets,
            roty_weights,
            dim_targets,
            dim_weights,
            delta_2dc_targets,
            delta_2dc_weights,
        )

    def loss(
        self,
        batch_size,
        depth_preds,
        depth_uncertainty_preds,
        dim_preds,
        alpha_preds,
        delta_2dc_preds,
        labels,
        depth_targets,
        depth_weights,
        roty_targets,
        roty_weights,
        dim_targets,
        dim_weights,
        delta_2dc_targets,
        delta_2dc_weights,
        reduction_override=None,
    ):
        losses = dict()

        pos_inds = labels > 0

        if depth_preds is not None and self.cfg.with_depth:
            depth_weights[depth_targets <= 0] = 0

            def get_depth_gt(gt, scale: float = 2.0):
                return torch.where(
                    gt > 0, torch.log(gt) * scale, -gt.new_ones(1)
                )

            if self.cfg.reg_class_agnostic:
                pos_depth_preds = depth_preds[pos_inds]
            else:
                pos_depth_preds = depth_preds.view(batch_size, -1, 1)[
                    pos_inds, labels[pos_inds]
                ]

            pos_depth_targets = get_depth_gt(depth_targets[pos_inds])
            pos_depth_weights = depth_weights[pos_inds]

            losses["loss_depth"] = self.loss_depth(
                pos_depth_preds, pos_depth_targets, weight=pos_depth_weights
            )

            if depth_uncertainty_preds is not None and self.with_uncertainty:
                pos_depth_self_labels = torch.exp(
                    -torch.abs(pos_depth_preds - pos_depth_targets) * 5.0
                )

                pos_depth_self_weights = torch.where(
                    pos_depth_self_labels > 0.8,
                    pos_depth_weights.new_ones(1) * 5.0,
                    pos_depth_weights.new_ones(1) * 0.1,
                )

                if self.cfg.reg_class_agnostic:
                    pos_depth_uncertainty_preds = depth_uncertainty_preds[
                        pos_inds
                    ]
                else:
                    pos_depth_uncertainty_preds = depth_uncertainty_preds.view(
                        batch_size, -1, 1
                    )[pos_inds, labels[pos_inds]]

                losses["loss_unc"] = self.loss_uncertainty(
                    pos_depth_uncertainty_preds,
                    pos_depth_self_labels.detach().clone(),
                    pos_depth_self_weights,
                    reduction_override=reduction_override,
                )
                losses["unc_acc"] = accuracy(
                    torch.cat(
                        [
                            1.0 - pos_depth_uncertainty_preds[:, None],
                            pos_depth_uncertainty_preds[:, None],
                        ],
                        dim=1,
                    ),
                    (pos_depth_self_labels > 0.8).detach().clone(),
                )
        if dim_preds is not None and self.with_dim:
            dim_weights[dim_targets <= 0] = 0

            def get_dim_gt(gt, scale: float = 2.0):
                return torch.where(
                    gt > 0, torch.log(gt) * scale, gt.new_ones(1)
                )

            if self.reg_class_agnostic:
                pos_dim_preds = dim_preds[pos_inds]
            else:
                pos_dim_preds = dim_preds.view(batch_size, -1, 3)[
                    pos_inds, labels[pos_inds]
                ]
            pos_dim_targets = get_dim_gt(dim_targets[pos_inds])
            pos_dim_weights = dim_weights[pos_inds]
            losses["loss_dim"] = self.loss_dim(
                pos_dim_preds, pos_dim_targets, weight=pos_dim_weights
            )

        if alpha_preds is not None and self.with_rot:
            alpha_weights[alpha_targets <= -10] = 0

            def get_rot_bin_gt(alpha_targets):
                bin_cls = alpha_targets.new_zeros(
                    (len(alpha_targets), 2)
                ).long()
                bin_res = alpha_targets.new_zeros(
                    (len(alpha_targets), 2)
                ).float()

                for i in range(len(alpha_targets)):
                    if (
                        alpha_targets[i] < np.pi / 6.0
                        or alpha_targets[i] > 5 * np.pi / 6.0
                    ):
                        bin_cls[i, 0] = 1
                        bin_res[i, 0] = alpha_targets[i] - (-0.5 * np.pi)

                    if (
                        alpha_targets[i] > -np.pi / 6.0
                        or alpha_targets[i] < -5 * np.pi / 6.0
                    ):
                        bin_cls[i, 1] = 1
                        bin_res[i, 1] = alpha_targets[i] - (0.5 * np.pi)
                return bin_cls, bin_res

            if self.reg_class_agnostic:
                pos_alpha_preds = alpha_preds[pos_inds].squeeze(1)
            else:
                pos_alpha_preds = alpha_preds.view(batch_size, -1, 8)[
                    pos_inds, labels[pos_inds]
                ]
            pos_rot_cls, pos_rot_res = get_rot_bin_gt(alpha_targets[pos_inds])
            pos_rot_weights = alpha_weights[pos_inds]
            avg_factor = max(
                torch.sum(pos_rot_weights > 0).float().item(), 1.0
            )
            losses["loss_alpha"] = self.loss_rot(
                pos_alpha_preds,
                pos_rot_cls,
                pos_rot_res,
                weight=pos_rot_weights,
                avg_factor=avg_factor,
            )

        if delta_2dc_preds is not None and self.with_2dc:
            pos_2dc_weights = delta_2dc_weights[pos_inds]

            def get_2dc_gt(gt_cen, scale: float = 10.0):
                return gt_cen / scale

            if self.reg_class_agnostic:
                pos_2dc_pred = delta_2dc_preds[pos_inds]
            else:
                pos_2dc_pred = delta_2dc_preds.view(batch_size, -1, 2)[
                    pos_inds, labels[pos_inds]
                ]
            pos_2dc_targets = get_2dc_gt(
                delta_2dc_targets[pos_inds], scale=self.center_scale
            )
            losses["loss_2dc"] = self.loss_2dc(
                pos_2dc_pred, pos_2dc_targets, weight=pos_2dc_weights
            )

        return losses
