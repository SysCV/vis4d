"""QD-3DT detector."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from vis4d.op.box.encoder import BoxEncoder3D, QD3DTBox3DEncoder
from vis4d.op.box.matchers import Matcher, MaxIoUMatcher
from vis4d.op.box.poolers import MultiScaleRoIAlign, RoIPooler
from vis4d.op.box.samplers import (
    CombinedSampler,
    Sampler,
    match_and_sample_proposals,
)
from vis4d.op.layer import add_conv_branch
from vis4d.op.geometry.rotation import generate_rotation_output

import pdb


class QD3DTBBox3DHeadOutput(NamedTuple):
    """Output of QD-3DT bounding box 3D head."""

    boxes_3d: List[torch.Tensor]  # (N, 12) x,y,z,h,w,l,rx,ry,rz,vx,vy,vz
    depth_uncertainty: List[torch.Tensor]  # (N, 1)


def get_default_proposal_pooler() -> RoIPooler:
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


def get_default_box_encoder() -> QD3DTBox3DEncoder:
    """Get the default bounding box encoder of QD-3DT bounding box 3D head."""
    return QD3DTBox3DEncoder()


class QD3DTBBox3DHead(nn.Module):
    """This class implements the QD-3DT bounding box 3D head."""

    def __init__(
        self,
        num_classes: int,
        proposal_pooler: None | RoIPooler = None,
        box_matcher: None | Matcher = None,
        box_sampler: None | Sampler = None,
        box_encoder: None | BoxEncoder3D = None,
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
        norm: Optional[str] = None,
        num_groups: int = 32,
        num_rotation_bins: int = 2,
        in_features: Tuple[str, ...] = ("p2", "p3", "p4", "p5"),
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
            box_encoder
            if box_encoder is not None
            else get_default_box_encoder()
        )
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_rotation_bins = num_rotation_bins
        self.in_features = in_features
        self.proposal_append_gt = proposal_append_gt
        self.cls_out_channels = num_classes

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
        module_lists = [self.shared_fcs]
        module_lists += [self.fc_dep_uncer]
        module_lists += [self.fc_dep, self.dep_fcs]
        module_lists += [self.fc_dim, self.dim_fcs]
        module_lists += [self.fc_rot, self.rot_fcs]
        module_lists += [self.fc_cen_2d, self.cen_2d_fcs]

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
        conv_out_dim: int,
        fc_out_dim: int,
        conv_has_bias: bool,
        norm: Optional[str],
        num_groups: int,
        is_shared: bool = False,
    ) -> Tuple[nn.ModuleList, nn.ModuleList, int]:
        """Init modules of head."""
        last_layer_dim = in_channels
        # add branch specific conv layers
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
                last_layer_dim *= np.prod(self.proposal_pooler.resolution)
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
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self,
        x_dep: torch.Tensor,
        x_dim: torch.Tensor,
        x_rot: torch.Tensor,
        x_cen_2d: torch.Tensor,
    ) -> torch.Tensor:
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
        self,
        features: List[torch.Tensor],
        boxes: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Get 3D bounding box prediction parameters."""
        roi_feats = self.proposal_pooler(features[2:6], boxes)
        x_dep, x_dim, x_rot, x_cen_2d = self.get_embeds(roi_feats)

        outputs = self.get_outputs(x_dep, x_dim, x_rot, x_cen_2d)

        outputs: List[torch.Tensor] = self.get_outputs(
            x_dep, x_dim, x_rot, x_cen_2d
        ).split([len(b) for b in boxes])
        return outputs

    def get_targets(
        self,
        pos_assigned_gt_inds: List[torch.Tensor],
        targets: LabelInstances,
        cam_intrinsics: Intrinsics,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get 3D bounding box targets for training."""
        bbox_targets = self.bbox_coder.encode(
            targets.boxes2d, targets.boxes3d, cam_intrinsics
        )

        bbox_targets = [
            b[p] for b, p in zip(bbox_targets, pos_assigned_gt_inds)
        ]

        labels = [
            t.class_ids[p]
            for t, p in zip(targets.boxes2d, pos_assigned_gt_inds)
        ]
        return bbox_targets, labels

    # def forward_train(
    #     self,
    #     inputs: InputSample,
    #     features: FeatureMaps,
    #     boxes: List[Boxes2D],
    #     targets: LabelInstances,
    # ) -> Tuple[LossesType, SamplingResult]:
    #     """Forward pass during training stage.

    #     Args:
    #         inputs: InputSamples (images, metadata, etc). Batched.
    #         features: Input feature maps. Batched.
    #         boxes: Input boxes to apply RoIHead on.
    #         targets: Targets corresponding to InputSamples.

    #     Returns:
    #         LossesType: A dict of scalar loss tensors.
    #         SamplingResult: Sampling result.
    #     """
    #     assert features is not None, "QD-3DT box3D head requires features!"
    #     features_list = [features[f] for f in self.in_features]

    #     # match and sample
    #     sampling_results = match_and_sample_proposals(
    #         self.matcher,
    #         self.sampler,
    #         boxes,
    #         inputs.targets.boxes2d,
    #         self.proposal_append_gt,
    #     )
    #     positives = [l == 1 for l in sampling_results.sampled_labels]
    #     pos_assigned_gt_inds = [
    #         i[p] if len(p) != 0 else p
    #         for i, p in zip(sampling_results.sampled_target_indices, positives)
    #     ]
    #     pos_boxes = [
    #         b[p] for b, p in zip(sampling_results.sampled_boxes, positives)
    #     ]

    #     predictions = self.get_predictions(features_list, pos_boxes)

    #     tgt_params, labels = self.get_targets(
    #         pos_assigned_gt_inds, targets, inputs.intrinsics
    #     )
    #     loss = self.loss(
    #         torch.cat(predictions), torch.cat(tgt_params), torch.cat(labels)
    #     )
    #     return loss, sampling_results

    def forward(
        self,
        features: list[Tensor],
        boxes: list[Tensor],
        class_ids: list[Tensor],
        intrinsics: torch.Tensor,
    ) -> QD3DTBBox3DHeadOutput:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            List[Boxes3D]: Prediction output.
        """
        assert features is not None, "QD-3DT box3D head requires features!"
        if sum(len(b) for b in boxes) == 0:
            boxes_3d = [
                torch.empty((0, 12), device=boxes[0].device)
                for _ in range(len(boxes))
            ]
            depth_uncertainty = [
                torch.empty((0, 1), device=boxes[0].device)
                for _ in range(len(boxes))
            ]
            return QD3DTBBox3DHeadOutput(
                boxes_3d=boxes_3d,
                depth_uncertainty=depth_uncertainty,
            )

        predictions = self.get_predictions(features, boxes)

        boxes_3d = []
        depth_uncertainty = []
        for (boxes_, class_ids_, box_deltas_, intrinsics_) in zip(
            boxes, class_ids, predictions, intrinsics
        ):
            if len(boxes_) == 0:
                boxes_3d.append(torch.empty(0, 12).to(boxes_.device))
                depth_uncertainty.append(torch.empty(0, 1).to(boxes_.device))
                continue
            box_deltas_ = box_deltas_[
                torch.arange(box_deltas_.shape[0]), class_ids_
            ]
            boxes3d_ = self.box_encoder.decode(
                boxes_, box_deltas_, intrinsics_
            )
            boxes_3d.append(boxes3d_[:, :12])
            depth_uncertainty.append(boxes3d_[:, 12])

        return QD3DTBBox3DHeadOutput(
            boxes_3d=boxes_3d,
            depth_uncertainty=depth_uncertainty,
        )

    def __call__(
        self,
        features: list[Tensor],
        boxes: list[Tensor],
        class_ids: list[Tensor],
        intrinsics: torch.Tensor,
    ) -> QD3DTBBox3DHeadOutput:
        """Type definition."""
        return self._call_impl(features, boxes, class_ids, intrinsics)
