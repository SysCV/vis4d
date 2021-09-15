"""3D Box Head definition for QD-3DT."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from vist.common.bbox.matchers import MatcherConfig, build_matcher
from vist.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from vist.common.bbox.samplers import (
    SamplerConfig,
    SamplingResult,
    build_sampler,
    match_and_sample_proposals,
)
from vist.common.layers import add_conv_branch
from vist.model.losses import LossConfig, build_loss
from vist.struct import Boxes2D, Boxes3D, InputSample, LossesType

from .base import BaseRoIHead, BaseRoIHeadConfig


class QD3DTBBox3DHeadConfig(BaseRoIHeadConfig):
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
    center_scale: float
    reg_class_agnostic: bool
    conv_has_bias: bool
    norm: str
    num_groups: int = 32
    with_depth: bool
    with_uncertainty: bool
    uncertainty_thres: float
    with_dim: bool
    with_rot: bool
    with_2dc: bool

    loss_3d: LossConfig

    proposal_append_gt: bool
    in_features: List[str] = ["p2", "p3", "p4", "p5"]
    proposal_pooler: RoIPoolerConfig
    proposal_sampler: SamplerConfig
    proposal_matcher: MatcherConfig


class QD3DTBBox3DHead(  # pylint: disable=too-many-instance-attributes
    BaseRoIHead
):
    """QD-3DT 3D Bounding Box Head."""

    def __init__(self, cfg: BaseRoIHeadConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QD3DTBBox3DHeadConfig(**cfg.dict())

        self.cls_out_channels = self.cfg.num_classes

        self.sampler = build_sampler(self.cfg.proposal_sampler)
        self.matcher = build_matcher(self.cfg.proposal_matcher)
        self.roi_pooler = build_roi_pooler(self.cfg.proposal_pooler)

        self.bbox_coder = Box3DCoder()  # TODO init from new module

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
        convs, last_layer_dim = add_conv_branch(
            num_branch_convs,
            in_channels,
            self.cfg.conv_out_dim,
            self.cfg.conv_has_bias,
            self.cfg.norm,
            self.cfg.num_groups,
        )

        fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if is_shared or self.cfg.num_shared_fcs == 0:
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

    def get_embeds(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate embedding from bbox feature."""
        # shared part
        if self.cfg.num_shared_convs > 0:
            for conv in self.shared_convs:
                feat = conv(feat)

        if self.cfg.num_shared_fcs > 0:
            feat = feat.view(feat.size(0), -1)
            for fc in self.shared_fcs:
                feat = self.relu(fc(feat))

        # separate branches
        x_dep = feat
        x_dim = feat
        x_rot = feat
        x_2dc = feat

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

    def get_logits(
        self,
        x_dep: torch.Tensor,
        x_dim: torch.Tensor,
        x_rot: torch.Tensor,
        x_2dc: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Generate logits for bbox 3d."""

        def get_rot(pred: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        features_list: List[torch.Tensor],
        pos_proposals: List[Boxes2D],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        pos_assigned_gt_inds: torch.Tensor,
        gt_bboxes_2d: Boxes2D,
        gt_bboxes_3d: Boxes3D,
        cam_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Get single box3d target for training."""
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
        pos_assigned_gt_inds: List[torch.Tensor],
        gt_bboxes_2d: List[Boxes2D],
        gt_bboxes_3d: List[Boxes3D],
        cam_intrinsics: List[torch.Tensor],
        concat: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get box3d target for training."""
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
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        boxes: List[Boxes2D],
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            LossesType: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling result.
        """
        features_list = [features[f] for f in self.cfg.in_features]

        # match and sample
        sampling_results = match_and_sample_proposals(
            self.matcher,
            self.sampler,
            boxes,
            inputs.boxes2d,
            self.cfg.proposal_append_gt,
        )
        positives = [l == 1 for l in sampling_results.sampled_labels]
        pos_assigned_gt_inds = [
            i[p]
            for i, p in zip(sampling_results.sampled_target_indices, positives)
        ]
        pos_proposals = [
            b[p] for b, p in zip(sampling_results.sampled_boxes, positives)
        ]

        bbox_3d_preds, _ = self.forward(features_list, pos_proposals)

        cam_intrinsics = [
            inputs.intrinsics.tensor[i]
            for i in range(inputs.intrinsics.tensor.shape[0])
        ]
        bbox3d_targets, labels = self.get_targets(
            pos_proposals,
            pos_assigned_gt_inds,
            inputs.boxes2d,
            inputs.boxes3d,
            cam_intrinsics,
        )

        loss_bbox_3d = self.loss_3d(bbox_3d_preds, bbox3d_targets, labels)

        return loss_bbox_3d, sampling_results

    def forward_test(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        boxes: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[Boxes3D]]]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            List[LabelInstance]: Prediction output.
        """
        features_list = [features[f] for f in self.cfg.in_features]
        bbox_3d_preds, _ = self.forward(features_list, boxes)

        return self.bbox_coder.decode(
            boxes[0],
            bbox_3d_preds,
            inputs.intrinsics.tensor[0],
            self.cfg.with_uncertainty,
            self.cfg.uncertainty_thres,
        )
