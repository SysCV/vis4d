"""QD-3DT detector."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from vis4d.common.typing import LossesType
from vis4d.op.box.encoder.qd_3dt import QD3DTBox3DDecoder, QD3DTBox3DEncoder
from vis4d.op.box.matchers import Matcher, MaxIoUMatcher
from vis4d.op.box.poolers import MultiScaleRoIAlign, MultiScaleRoIPooler
from vis4d.op.box.samplers import (
    CombinedSampler,
    Sampler,
    match_and_sample_proposals,
)
from vis4d.op.geometry.rotation import generate_rotation_output
from vis4d.op.layer import Conv2d, add_conv_branch
from vis4d.op.layer.weight_init import kaiming_init, xavier_init
from vis4d.op.loss.base import Loss
from vis4d.op.loss.common import rotation_loss, smooth_l1_loss
from vis4d.op.loss.reducer import LossReducer, SumWeightedLoss, mean_loss


class QD3DTBBox3DHeadOutput(NamedTuple):
    """QD-3DT bounding box 3D head training output."""

    predictions: list[Tensor]
    targets: Tensor | None
    labels: Tensor | None


class QD3DTDet3DOut(NamedTuple):
    """Output of QD-3DT bounding box 3D head.

    Attributes:
        boxes_3d (list[Tensor]): Predicted 3D bounding boxes. Each tensor has
            shape (N, 12) and contains x,y,z,h,w,l,rx,ry,rz,vx,vy,vz.
        depth_uncertainty (list[Tensor]): Predicted depth uncertainty. Each
            tensor has shape (N, 1).
    """

    boxes_3d: list[Tensor]
    depth_uncertainty: list[Tensor]


def get_default_proposal_pooler() -> MultiScaleRoIAlign:
    """Get default proposal pooler of QD-3DT bounding box 3D head."""
    return MultiScaleRoIAlign(
        resolution=[7, 7], strides=[4, 8, 16, 32], sampling_ratio=0
    )


def get_default_box_sampler() -> CombinedSampler:
    """Get default box sampler of QD-3DT bounding box 3D head."""
    return CombinedSampler(
        batch_size=512,
        positive_fraction=0.25,
        pos_strategy="instance_balanced",
        neg_strategy="iou_balanced",
    )


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default box matcher of QD-3DT bounding box 3D head."""
    return MaxIoUMatcher(
        thresholds=[0.5, 0.5],
        labels=[0, -1, 1],
        allow_low_quality_matches=False,
    )


def get_default_box_codec(
    center_scale: float = 10.0,
    depth_log_scale: float = 2.0,
    dim_log_scale: float = 2.0,
    num_rotation_bins: int = 2,
    bin_overlap: float = 1 / 6,
) -> tuple[QD3DTBox3DEncoder, QD3DTBox3DDecoder]:
    """Get the default bounding box encoder and decoder."""
    return (
        QD3DTBox3DEncoder(
            center_scale=center_scale,
            depth_log_scale=depth_log_scale,
            dim_log_scale=dim_log_scale,
            num_rotation_bins=num_rotation_bins,
            bin_overlap=bin_overlap,
        ),
        QD3DTBox3DDecoder(
            center_scale=center_scale,
            depth_log_scale=depth_log_scale,
            dim_log_scale=dim_log_scale,
            num_rotation_bins=num_rotation_bins,
        ),
    )


class QD3DTBBox3DHead(nn.Module):
    """This class implements the QD-3DT bounding box 3D head."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments, line-too-long
        self,
        num_classes: int,
        proposal_pooler: None | MultiScaleRoIPooler = None,
        box_matcher: None | Matcher = None,
        box_sampler: None | Sampler = None,
        box_encoder: None | QD3DTBox3DEncoder = None,
        proposal_append_gt: bool = True,
        num_shared_convs: int = 2,
        num_shared_fcs: int = 0,
        num_dep_convs: int = 4,
        num_dep_fcs: int = 0,
        num_dim_convs: int = 4,
        num_dim_fcs: int = 0,
        num_rot_convs: int = 4,
        num_rot_fcs: int = 0,
        num_cen_2d_convs: int = 4,
        num_cen_2d_fcs: int = 0,
        in_channels: int = 256,
        conv_out_dim: int = 256,
        fc_out_dim: int = 1024,
        roi_feat_size: int = 7,
        conv_has_bias: bool = True,
        norm: None | str = None,
        num_groups: int = 32,
        num_rotation_bins: int = 2,
        start_level: int = 2,
    ):
        """Initialize the QD-3DT bounding box 3D head."""
        super().__init__()
        self.proposal_pooler = (
            proposal_pooler
            if proposal_pooler is not None
            else get_default_proposal_pooler()
        )
        self.box_matcher = (
            box_matcher
            if box_matcher is not None
            else get_default_box_matcher()
        )
        self.box_sampler = (
            box_sampler
            if box_sampler is not None
            else get_default_box_sampler()
        )
        self.box_encoder = (
            box_encoder if box_encoder is not None else QD3DTBox3DEncoder()
        )
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_rotation_bins = num_rotation_bins
        self.proposal_append_gt = proposal_append_gt
        self.cls_out_channels = num_classes

        # Used feature layers are [start_level, end_level)
        self.start_level = start_level
        num_strides = len(self.proposal_pooler.scales)
        self.end_level = start_level + num_strides

        # add shared convs and fcs
        (
            self.shared_convs,
            self.shared_fcs,
            self.shared_out_channels,
        ) = self._add_conv_fc_branch(
            num_shared_convs,
            num_shared_fcs,
            in_channels,
            conv_out_dim,
            fc_out_dim,
            conv_has_bias,
            norm,
            num_groups,
            True,
        )

        # add depth specific branch
        (
            self.dep_convs,
            self.dep_fcs,
            self.dep_last_dim,
        ) = self._add_conv_fc_branch(
            num_dep_convs,
            num_dep_fcs,
            self.shared_out_channels,
            conv_out_dim,
            fc_out_dim,
            conv_has_bias,
            norm,
            num_groups,
        )

        # add dim specific branch
        (
            self.dim_convs,
            self.dim_fcs,
            self.dim_last_dim,
        ) = self._add_conv_fc_branch(
            num_dim_convs,
            num_dim_fcs,
            self.shared_out_channels,
            conv_out_dim,
            fc_out_dim,
            conv_has_bias,
            norm,
            num_groups,
        )

        # add rot specific branch
        (
            self.rot_convs,
            self.rot_fcs,
            self.rot_last_dim,
        ) = self._add_conv_fc_branch(
            num_rot_convs,
            num_rot_fcs,
            self.shared_out_channels,
            conv_out_dim,
            fc_out_dim,
            conv_has_bias,
            norm,
            num_groups,
        )

        # add delta 2D center specific branch
        (
            self.cen_2d_convs,
            self.cen_2d_fcs,
            self.cen_2d_last_dim,
        ) = self._add_conv_fc_branch(
            num_cen_2d_convs,
            num_cen_2d_fcs,
            self.shared_out_channels,
            conv_out_dim,
            fc_out_dim,
            conv_has_bias,
            norm,
            num_groups,
        )

        if num_shared_fcs == 0:
            if num_dep_fcs == 0:
                self.dep_last_dim *= roi_feat_size * roi_feat_size
            if num_dim_fcs == 0:
                self.dim_last_dim *= roi_feat_size * roi_feat_size
            if num_rot_fcs == 0:
                self.rot_last_dim *= roi_feat_size * roi_feat_size
            if num_cen_2d_fcs == 0:
                self.cen_2d_last_dim *= roi_feat_size * roi_feat_size

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        out_dim_dep = self.cls_out_channels
        self.fc_dep = nn.Linear(self.dep_last_dim, out_dim_dep)

        self.fc_dep_uncer = nn.Linear(self.dep_last_dim, out_dim_dep)

        out_dim_size = 3 * self.cls_out_channels
        self.fc_dim = nn.Linear(self.dim_last_dim, out_dim_size)

        out_rot_size = 3 * num_rotation_bins * self.cls_out_channels
        self.fc_rot = nn.Linear(self.rot_last_dim, out_rot_size)

        out_cen_2d_size = 2 * self.cls_out_channels
        self.fc_cen_2d = nn.Linear(self.cen_2d_last_dim, out_cen_2d_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Init weights of modules in head."""
        module_lists: list[nn.ModuleList | nn.Linear | Conv2d] = []
        module_lists += [self.shared_convs]
        module_lists += [self.shared_fcs]
        module_lists += [self.dep_convs]
        module_lists += [self.fc_dep_uncer]
        module_lists += [self.fc_dep, self.dep_fcs]
        module_lists += [self.dim_convs]
        module_lists += [self.fc_dim, self.dim_fcs]
        module_lists += [self.rot_convs]
        module_lists += [self.fc_rot, self.rot_fcs]
        module_lists += [self.cen_2d_convs]
        module_lists += [self.fc_cen_2d, self.cen_2d_fcs]

        for module_list in module_lists:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    xavier_init(m, distribution="uniform")
                elif isinstance(m, Conv2d):
                    kaiming_init(m)

    def _add_conv_fc_branch(
        self,
        num_branch_convs: int,
        num_branch_fcs: int,
        in_channels: int,
        conv_out_dim: int,
        fc_out_dim: int,
        conv_has_bias: bool,
        norm: None | str,
        num_groups: int,
        is_shared: bool = False,
    ) -> tuple[nn.ModuleList, nn.ModuleList, int]:
        """Init modules of head."""
        convs, last_layer_dim = add_conv_branch(
            num_branch_convs,
            in_channels,
            conv_out_dim,
            conv_has_bias,
            norm,
            num_groups,
        )

        fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if is_shared or num_branch_fcs == 0:
                last_layer_dim *= int(np.prod(self.proposal_pooler.resolution))
            for i in range(num_branch_fcs):
                fc_in_dim = last_layer_dim if i == 0 else fc_out_dim
                fcs.append(
                    nn.Sequential(
                        nn.Linear(fc_in_dim, fc_out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
            last_layer_dim = fc_out_dim
        return convs, fcs, last_layer_dim

    def get_embeds(
        self, feat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate embedding from bbox feature."""
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                feat = conv(feat)

        if self.num_shared_fcs > 0:
            feat = feat.view(feat.size(0), -1)
            for fc in self.shared_fcs:
                feat = self.relu(fc(feat))

        # separate branches
        x_dep = feat
        x_dim = feat
        x_rot = feat
        x_cen_2d = feat

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
            x_cen_2d = conv(x_cen_2d)
        if x_cen_2d.dim() > 2:
            x_cen_2d = x_cen_2d.view(x_cen_2d.size(0), -1)
        for fc in self.cen_2d_fcs:
            x_cen_2d = self.relu(fc(x_cen_2d))

        return x_dep, x_dim, x_rot, x_cen_2d

    def get_outputs(
        self, x_dep: Tensor, x_dim: Tensor, x_rot: Tensor, x_cen_2d: Tensor
    ) -> Tensor:
        """Generate output 3D bounding box parameters."""
        depth = self.fc_dep(x_dep).view(-1, self.cls_out_channels, 1)
        depth_uncertainty = self.fc_dep_uncer(x_dep).view(
            -1, self.cls_out_channels, 1
        )
        dim = self.fc_dim(x_dim).view(-1, self.cls_out_channels, 3)
        alpha = generate_rotation_output(
            self.fc_rot(x_rot), self.num_rotation_bins
        )
        delta_cen_2d = self.fc_cen_2d(x_cen_2d).view(
            -1, self.cls_out_channels, 2
        )
        return torch.cat(
            [delta_cen_2d, depth, dim, alpha, depth_uncertainty], -1
        )

    def get_predictions(
        self, features: list[Tensor], boxes_2d: list[Tensor]
    ) -> list[Tensor]:
        """Get 3D bounding box prediction parameters."""
        if sum(len(b) for b in boxes_2d) == 0:  # pragma: no cover
            return [
                torch.empty(
                    (
                        0,
                        self.cls_out_channels,
                        6 + 3 * self.num_rotation_bins + 1,
                    ),
                    device=boxes_2d[0].device,
                )
            ] * len(boxes_2d)

        roi_feats = self.proposal_pooler(
            features[self.start_level : self.end_level], boxes_2d
        )
        x_dep, x_dim, x_rot, x_cen_2d = self.get_embeds(roi_feats)

        outputs: list[Tensor] = list(
            self.get_outputs(x_dep, x_dim, x_rot, x_cen_2d).split(
                [len(b) for b in boxes_2d]
            )
        )
        return outputs

    def get_targets(
        self,
        pos_assigned_gt_inds: list[Tensor],
        target_boxes: list[Tensor],
        target_boxes3d: list[Tensor],
        target_class_ids: list[Tensor],
        intrinsics: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Get 3D bounding box targets for training."""
        targets = []
        labels = []
        for i, (tgt_boxes, tgt_boxes3d, intrinsics_) in enumerate(
            zip(target_boxes, target_boxes3d, intrinsics)
        ):
            bbox_target = self.box_encoder(tgt_boxes, tgt_boxes3d, intrinsics_)
            targets.append(bbox_target[pos_assigned_gt_inds[i]])

            labels.append(target_class_ids[i][pos_assigned_gt_inds[i]])

        return torch.cat(targets), torch.cat(labels)

    def forward(
        self,
        features: list[Tensor],
        det_boxes: list[Tensor],
        intrinsics: Tensor | None = None,
        target_boxes: list[Tensor] | None = None,
        target_boxes3d: list[Tensor] | None = None,
        target_class_ids: list[Tensor] | None = None,
    ) -> QD3DTBBox3DHeadOutput:
        """Forward."""
        if (
            intrinsics is not None
            and target_boxes is not None
            and target_boxes3d is not None
            and target_class_ids is not None
        ):
            if self.proposal_append_gt:
                det_boxes = [
                    torch.cat([d, t]) for d, t in zip(det_boxes, target_boxes)
                ]

            (
                sampled_box_indices,
                sampled_target_indices,
                sampled_labels,
            ) = match_and_sample_proposals(
                self.box_matcher, self.box_sampler, det_boxes, target_boxes
            )
            positives = [torch.eq(l, 1) for l in sampled_labels]
            pos_assigned_gt_inds = [
                i[p] if len(p) != 0 else p
                for i, p in zip(sampled_target_indices, positives)
            ]
            pos_boxes = [
                b[s_i][p]
                for b, s_i, p in zip(det_boxes, sampled_box_indices, positives)
            ]
            predictions = self.get_predictions(features, pos_boxes)

            targets, labels = self.get_targets(
                pos_assigned_gt_inds,
                target_boxes,
                target_boxes3d,
                target_class_ids,
                intrinsics,
            )

            return QD3DTBBox3DHeadOutput(
                predictions=predictions, targets=targets, labels=labels
            )

        predictions = self.get_predictions(features, det_boxes)

        return QD3DTBBox3DHeadOutput(predictions, None, None)

    def __call__(
        self,
        features: list[Tensor],
        det_boxes: list[Tensor],
        intrinsics: Tensor | None = None,
        target_boxes: list[Tensor] | None = None,
        target_boxes3d: list[Tensor] | None = None,
        target_class_ids: list[Tensor] | None = None,
    ) -> QD3DTBBox3DHeadOutput:
        """Type definition."""
        return self._call_impl(
            features,
            det_boxes,
            intrinsics,
            target_boxes,
            target_boxes3d,
            target_class_ids,
        )


class RoI2Det3D:
    """Post processing for QD3DTBBox3DHead."""

    def __init__(self, box_decoder: None | QD3DTBox3DDecoder = None) -> None:
        """Initialize."""
        self.box_decoder = (
            QD3DTBox3DDecoder() if box_decoder is None else box_decoder
        )

    def __call__(
        self,
        predictions: list[Tensor],
        boxes_2d: list[Tensor],
        class_ids: list[Tensor],
        intrinsics: Tensor,
    ) -> QD3DTDet3DOut:
        """Forward pass during testing stage.

        Args:
            predictions(list[Tensor]): Predictions.
            boxes_2d(list[Tensor]): 2D boxes.
            class_ids(list[Tensor]): Class IDs.
            intrinsics(Tensor): Camera intrinsics.

        Returns:
            QD3DTDet3DOut: QD3DT 3D detection output.
        """
        boxes_3d = []
        depth_uncertainty = []
        device = boxes_2d[0].device
        for _boxes_2d, _class_ids, _boxes_deltas, _intrinsics in zip(
            boxes_2d, class_ids, predictions, intrinsics
        ):
            if len(_boxes_2d) == 0:
                boxes_3d.append(torch.empty(0, 12).to(device))
                depth_uncertainty.append(torch.empty(0).to(device))
                continue

            _boxes_deltas = _boxes_deltas[
                torch.arange(_boxes_deltas.shape[0]), _class_ids
            ]

            depth_uncertainty.append(
                _boxes_deltas[:, -1].clamp(min=0.0, max=1.0)
            )
            boxes_3d.append(
                self.box_decoder(_boxes_2d, _boxes_deltas, _intrinsics)
            )

        return QD3DTDet3DOut(
            boxes_3d=boxes_3d, depth_uncertainty=depth_uncertainty
        )


class Box3DUncertaintyLoss(Loss):
    """Box3d loss for QD-3DT."""

    def __init__(
        self,
        reducer: LossReducer = mean_loss,
        center_loss_weight: float = 1.0,
        depth_loss_weight: float = 1.0,
        dimension_loss_weight: float = 1.0,
        rotation_loss_weight: float = 1.0,
        uncertainty_loss_weight: float = 1.0,
        num_rotation_bins: int = 2,
    ) -> None:
        """Creates an instance of the class.

        Args:
            reducer (LossReducer): Reducer for the loss function.
            center_loss_weight (float): Weight for center loss.
            depth_loss_weight (float): Weight for depth loss.
            dimension_loss_weight (float): Weight for dimension loss.
            rotation_loss_weight (float): Weight for rotation loss.
            uncertainty_loss_weight (float): Weight for uncertainty loss.
            num_rotation_bins (int): Number of rotation bins.
        """
        super().__init__(reducer)
        self.center_loss_weight = center_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.dimension_loss_weight = dimension_loss_weight
        self.rotation_loss_weight = rotation_loss_weight
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.num_rotation_bins = num_rotation_bins

    def forward(
        self, pred: Tensor, target: Tensor, labels: Tensor
    ) -> LossesType:
        """Compute box3d loss.

        Args:
            pred (Tensor): Box predictions of shape [N, num_classes,
                6 + 3 * num_rotations_bins].
            target (torcch.Tensor): Target boxes of shape [N,
                6 + num_rotation_bins].
            labels (Tensor): Target Labels of shape [N].

        Returns:
           dict[str, Tensor] containing 'delta 2dc', 'dimension', 'depth',
             'rotation' and 'uncertainty' loss.
        """
        if pred.size(0) == 0:
            loss_ctr3d = loss_dep3d = loss_dim3d = loss_rot3d = loss_conf3d = (
                pred.sum() * 0
            )
            result_dict = {
                "loss_ctr3d": loss_ctr3d,
                "loss_dep3d": loss_dep3d,
                "loss_dim3d": loss_dim3d,
                "loss_rot3d": loss_rot3d,
                "loss_conf3d": loss_conf3d,
            }

            return result_dict

        pred = pred[torch.arange(pred.shape[0], device=pred.device), labels]

        # delta 2dc loss
        loss_cen = smooth_l1_loss(
            pred[:, :2], target[:, :2], reducer=self.reducer, beta=1 / 9
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
            reducer=self.reducer,
        )

        # uncertainty loss
        pos_depth_self_labels = torch.exp(
            -torch.mul(torch.abs(pred[:, 2] - target[:, 2]), 5.0)
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
                pos_depth_self_weights, len(pos_depth_self_weights)
            ),
            beta=1 / 9,
        )

        return {
            "loss_ctr3d": torch.mul(self.center_loss_weight, loss_cen),
            "loss_dep3d": torch.mul(self.depth_loss_weight, loss_dep),
            "loss_dim3d": torch.mul(self.dimension_loss_weight, loss_dim),
            "loss_rot3d": torch.mul(self.rotation_loss_weight, loss_rot),
            "loss_unc3d": torch.mul(self.uncertainty_loss_weight, loss_unc3d),
        }
