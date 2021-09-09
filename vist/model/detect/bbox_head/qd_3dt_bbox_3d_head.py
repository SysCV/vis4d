"""Similarity Head definition for quasi-dense instance similarity learning."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vist.common.bbox.utils import Box3DCoder
from vist.common.bbox.matchers import MatcherConfig, build_matcher
from vist.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler

from vist.common.bbox.samplers import (
    SamplerConfig,
    SamplingResult,
    build_sampler,
)
from vist.common.layers import Conv2d
from vist.model.losses import LossConfig, build_loss
from vist.model.detect.mmdet_utils import proposals_to_mmdet
from vist.struct import Boxes2D, Boxes3D

from .base import BaseBoundingBoxConfig, BaseBoundingBoxHead
import pdb


class QD3DTBBox3DHeadConfig(BaseBoundingBoxConfig):
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
    num_groups: int = 32
    with_depth: bool
    with_uncertainty: bool
    with_dim: bool
    with_rot: bool
    with_2dc: bool

    loss_3d: LossConfig

    proposal_append_gt: bool
    in_features: List[str] = ["p2", "p3", "p4", "p5"]
    proposal_pooler: RoIPoolerConfig
    proposal_sampler: SamplerConfig
    proposal_matcher: MatcherConfig


class QD3DTBBox3DHead(BaseBoundingBoxHead):
    """QD-3DT 3D Bounding Box Head."""

    def __init__(self, cfg: BaseBoundingBoxConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QD3DTBBox3DHeadConfig(**cfg.dict())

        self.cls_out_channels = self.cfg.num_classes

        self.sampler = build_sampler(self.cfg.proposal_sampler)
        self.matcher = build_matcher(self.cfg.proposal_matcher)
        self.roi_pooler = build_roi_pooler(self.cfg.proposal_pooler)

        self.bbox_coder = Box3DCoder()

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
        self.loss_3d = build_loss(self.cfg.loss_3d)

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
        norm = getattr(nn, self.cfg.norm)
        if norm == nn.GroupNorm:
            norm = lambda x: nn.GroupNorm(self.cfg.num_groups, x)
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
                        norm=norm(self.cfg.conv_out_dim),
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

    @torch.no_grad()  # type: ignore
    def match_and_sample_proposals(
        self, proposals: List[Boxes2D], targets: List[Boxes2D]
    ) -> SamplingResult:
        """Match proposals to targets and subsample."""
        if self.cfg.proposal_append_gt:
            proposals = [
                Boxes2D.cat([p, t]) for p, t in zip(proposals, targets)
            ]
        matching = self.matcher.match(proposals, targets)
        return self.sampler.sample(matching, proposals, targets)

    def forward(
        self,
        features_list: List[torch.Tensor],
        pos_proposals: List[Boxes2D],
    ):
        """Forward bbox 3d estimation."""
        roi_feats = self.roi_pooler.pool(features_list, pos_proposals)

        x_dep, x_dim, x_rot, x_2dc = self.get_embeds(roi_feats)
        (
            depth_preds,
            depth_uncertainty_preds,
            dim_preds,
            alpha_preds,
            delta_2dc_preds,
        ) = self.get_logits(x_dep, x_dim, x_rot, x_2dc)

        outputs = [
            delta_2dc_preds.view(-1, self.cfg.num_classes, 2),
            depth_preds.view(-1, self.cfg.num_classes, 1),
            dim_preds.view(-1, self.cfg.num_classes, 3),
            alpha_preds.view(-1, self.cfg.num_classes, 8),
            depth_uncertainty_preds.view(-1, self.cfg.num_classes, 1),
        ]

        bbox_3d_preds = torch.cat(outputs, -1)

        return bbox_3d_preds, roi_feats

    def _get_target_single(
        self,
        pos_proposals: Boxes2D,
        pos_assigned_gt_inds: torch.tensor,
        gt_bboxes_2d: Boxes2D,
        gt_bboxes_3d: Boxes3D,
        cam_intrinsics: torch.tensor,
    ):
        num_pos = pos_proposals.boxes.size(0)
        bbox_targets = pos_proposals.boxes.new_zeros(
            num_pos, 2 + 1 + 3 + 4
        )  # angle only 4 params in GT (2 bin, res cos/sin)

        if num_pos > 0:
            bbox_targets = self.bbox_coder.encode(
                gt_bboxes_2d[pos_assigned_gt_inds],
                gt_bboxes_3d[pos_assigned_gt_inds],
                cam_intrinsics,
            )
        return bbox_targets

    def get_targets(
        self,
        pos_proposals: List[Boxes2D],
        pos_assigned_gt_inds: List[torch.tensor],
        gt_bboxes_2d: List[Boxes2D],
        gt_bboxes_3d: List[Boxes3D],
        cam_intrinsics: List[torch.tensor],
        concat=True,
    ):
        bbox_targets = list(
            map(
                self._get_target_single,
                pos_proposals,
                pos_assigned_gt_inds,
                gt_bboxes_2d,
                gt_bboxes_3d,
                cam_intrinsics,
            )
        )

        labels = [
            t.class_ids[p] for t, p in zip(gt_bboxes_2d, pos_assigned_gt_inds)
        ]

        if concat:
            bbox_targets = torch.cat(bbox_targets, 0)
            labels = torch.cat(labels, 0)

        return bbox_targets, labels

    def forward_train(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
        targets_3d: Optional[List[Boxes3D]] = None,
        cam_intrinsics: Optional[torch.tensor] = None,
    ):
        """Forward pass during training stage.

        Args:
            features:
            proposals:
            targets:
            targets_3d:
            cam_intrinsics:

        Returns:
            LossesType: A dict of scalar loss tensors.
        """
        features_list = [features[f] for f in self.cfg.in_features]

        # match and sample
        sampling_results = self.match_and_sample_proposals(proposals, targets)
        positives = [l == 1 for l in sampling_results.sampled_labels]
        pos_assigned_gt_inds = [
            i[p]
            for i, p in zip(sampling_results.sampled_target_indices, positives)
        ]
        pos_proposals = [
            b[p] for b, p in zip(sampling_results.sampled_boxes, positives)
        ]

        (bbox_3d_preds, _,) = self.forward(
            features_list,
            pos_proposals,
        )

        bbox3d_targets, labels = self.get_targets(
            pos_proposals,
            pos_assigned_gt_inds,
            targets,
            targets_3d,
            cam_intrinsics,
        )

        loss_bbox_3d = self.loss(bbox_3d_preds, bbox3d_targets, labels)

        return loss_bbox_3d

    def loss(self, bbox3d_pred, bbox_targets, labels):
        # if any positive boxes
        if not bbox3d_pred.size(0) == 0:
            losses = self.loss_3d(bbox3d_pred, bbox_targets, labels)
        else:
            loss_ctr3d = loss_dep3d = loss_dim3d = loss_rot3d = loss_conf3d = (
                bbox3d_pred.sum() * 0
            )
            losses = dict(
                loss_ctr3d=loss_ctr3d,
                loss_dep3d=loss_dep3d,
                loss_dim3d=loss_dim3d,
                loss_rot3d=loss_rot3d,
            )
            if self.with_confidence:
                losses["loss_conf3d"] = loss_conf3d

        return losses

    # TODO: Get rid of mmdet nms
    def get_det_bboxes(
        self,
        cls_score,
        bbox_2d_preds,
        bbox_3d_preds,
    ):
        """Transfer model prediction with nms into detection results."""
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if not self.cfg.reg_class_agnostic:
            bbox_2d_preds = bbox_2d_preds.view(scores.size(0), -1, 4)

        pdb.set_trace()

        bbox_2d_preds, bbox_3d_preds = self.bbox_coder.decode(
            bbox_2d_preds, bbox_3d_preds
        )

        return self.multiclass_3d_nms(
            bbox_2d_preds,
            scores,
            bbox_3d_preds,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
        )

    def multiclass_3d_nms(
        self,
        multi_bboxes,
        multi_scores,
        bbox_3d_preds,
        score_thr,
        nms_cfg,
        max_num=-1,
        score_factors=None,
    ):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the 0th column
                contains scores of the background class, but this will be ignored.
            bbox_3d_preds (Tensor): shape (n, #class*15) or (n, 15)
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_thr (float): NMS IoU threshold
            max_num (int): if there are more than max_num bboxes after NMS,
                only top max_num will be kept.
            score_factors (Tensor): The factors multiplied to scores before
                applying NMS

        Returns:
            tuple: (bbox_2d_preds, labels, bbox_3d_preds), tensors of shape
                (k, 5), (k, 1) and (k, 15). Labels are 0-based.
        """
        num_classes = multi_scores.size(1) - 1

        # exclude background category
        if self.cfg.reg_class_agnostic:
            bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
        else:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
        scores = multi_scores[:, 1:]

        # filter out boxes with low scores
        if self.cfg.with_uncertainty:
            valid_mask = (scores * depth_uncertainty_pred[:, 1:]) > score_thr
        else:
            valid_mask = scores > score_thr
        bboxes = bboxes[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]

        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
            depths = multi_bboxes.new_zeros((0,))
            depths_uncertainty = multi_bboxes.new_zeros((0,))
            dims = multi_bboxes.new_zeros((0, 3))
            rots = multi_bboxes.new_zeros((0,))
            cen_2ds = multi_bboxes.new_zeros((0, 2))
            return (
                bboxes,
                labels,
                depths,
                depths_uncertainty,
                dims,
                rots,
                cen_2ds,
            )

        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

        if depth_pred is not None:
            if reg_class_agnostic:
                depths = depth_pred.expand(-1, num_classes, -1)[valid_mask]
            else:
                depths = depth_pred.view(multi_scores.size(0), -1, 1)[:, 1:][
                    valid_mask
                ]
            depths = depths[keep[:max_num]]
        else:
            depths = None

        if depth_uncertainty_pred is not None:
            if reg_class_agnostic:
                depths_uncertainty = depth_uncertainty_pred.expand(
                    -1, num_classes, -1
                )[valid_mask]
            else:
                depths_uncertainty = depth_uncertainty_pred.view(
                    multi_scores.size(0), -1, 1
                )[:, 1:][valid_mask]
            depths_uncertainty = depths_uncertainty[keep[:max_num]]
        else:
            depths_uncertainty = None

        if dim_pred is not None:
            if reg_class_agnostic:
                dims = dim_pred.expand(-1, num_classes, -1)[valid_mask]
            else:
                dims = dim_pred.view(multi_scores.size(0), -1, 3)[:, 1:][
                    valid_mask
                ]
            dims = dims[keep[:max_num]]
        else:
            dims = None

        if rot_pred is not None:
            if reg_class_agnostic:
                rots = rot_pred.expand(-1, num_classes, -1)[valid_mask]
            else:
                rots = rot_pred.view(multi_scores.size(0), -1, 1)[:, 1:][
                    valid_mask
                ]
            rots = rots[keep[:max_num]]
        else:
            rots = None

        if cen_2d_pred is not None:
            if reg_class_agnostic:
                cen_2ds = cen_2d_pred.expand(-1, num_classes, -1)[valid_mask]
            else:
                cen_2ds = cen_2d_pred.view(multi_scores.size(0), -1, 2)[:, 1:][
                    valid_mask
                ]
            cen_2ds = cen_2ds[keep[:max_num]]
        else:
            cen_2ds = None

        return (
            dets[:max_num],
            labels[keep[:max_num]],
            depths,
            depths_uncertainty,
            dims,
            rots,
            cen_2ds,
        )
