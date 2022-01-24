"""3D Box Head definition for QD-3DT."""
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from vis4d.common.bbox.coders import BaseBoxCoder3D, QD3DTBox3DCoder
from vis4d.common.bbox.matchers import BaseMatcher
from vis4d.common.bbox.poolers import BaseRoIPooler
from vis4d.common.bbox.samplers import (
    BaseSampler,
    SamplingResult,
    match_and_sample_proposals,
)
from vis4d.common.geometry.rotation import generate_rotation_output
from vis4d.common.layers import add_conv_branch
from vis4d.common.module import build_module
from vis4d.model.losses import BaseLoss, Box3DUncertaintyLoss
from vis4d.struct import (
    Boxes2D,
    Boxes3D,
    FeatureMaps,
    InputSample,
    Intrinsics,
    LabelInstances,
    LossesType,
    ModuleCfg,
)

from .base import Det3DRoIHead


class QD3DTBBox3DHead(Det3DRoIHead):
    """QD-3DT 3D Bounding Box Head."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_classes: int,
        proposal_pooler: Union[ModuleCfg, BaseRoIPooler],
        proposal_sampler: Union[ModuleCfg, BaseSampler],
        proposal_matcher: Union[ModuleCfg, BaseMatcher],
        box3d_coder: Union[ModuleCfg, BaseBoxCoder3D] = QD3DTBox3DCoder(),
        loss: Union[ModuleCfg, BaseLoss] = Box3DUncertaintyLoss(),
        proposal_append_gt: bool = True,
        num_shared_convs: int = 2,
        num_shared_fcs: int = 0,
        num_dep_convs: int = 4,
        num_dep_fcs: int = 0,
        num_dim_convs: int = 4,
        num_dim_fcs: int = 0,
        num_rot_convs: int = 4,
        num_rot_fcs: int = 0,
        num_2dc_convs: int = 4,
        num_2dc_fcs: int = 0,
        in_channels: int = 256,
        conv_out_dim: int = 256,
        fc_out_dim: int = 1024,
        roi_feat_size: int = 7,
        conv_has_bias: bool = False,
        norm: Optional[str] = None,
        num_groups: int = 32,
        num_rotation_bins: int = 2,
        in_features: Tuple[str, ...] = ("p2", "p3", "p4", "p5"),
    ) -> None:
        """Init."""
        super().__init__()
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_rotation_bins = num_rotation_bins
        self.in_features = in_features
        self.proposal_append_gt = proposal_append_gt
        self.cls_out_channels = num_classes
        if isinstance(proposal_sampler, dict):
            self.sampler: BaseSampler = build_module(
                proposal_sampler, bound=BaseSampler
            )
        else:
            self.sampler = proposal_sampler

        if isinstance(proposal_matcher, dict):
            self.matcher: BaseMatcher = build_module(
                proposal_matcher, bound=BaseMatcher
            )
        else:
            self.matcher = proposal_matcher

        if isinstance(proposal_pooler, dict):
            self.roi_pooler: BaseRoIPooler = build_module(
                proposal_pooler, bound=BaseRoIPooler
            )
        else:
            self.roi_pooler = proposal_pooler

        if isinstance(box3d_coder, dict):
            self.bbox_coder: BaseBoxCoder3D = build_module(
                box3d_coder, bound=BaseBoxCoder3D
            )
        else:
            self.bbox_coder = box3d_coder

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

        # add 2dc specific branch
        (
            self.cen_2d_convs,
            self.cen_2d_fcs,
            self.cen_2d_last_dim,
        ) = self._add_conv_fc_branch(
            num_2dc_convs,
            num_2dc_fcs,
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
            if num_2dc_fcs == 0:
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

        out_2dc_size = 2 * self.cls_out_channels
        self.fc_2dc = nn.Linear(self.cen_2d_last_dim, out_2dc_size)

        self._init_weights()

        # losses
        if isinstance(loss, dict):
            self.loss: BaseLoss = build_module(loss, bound=BaseLoss)
        else:
            self.loss = loss

        assert (
            self.bbox_coder.num_rotation_bins == self.loss.num_rotation_bins
        ), (
            "num_rotation_bins must be consistent between head, loss and box "
            "coder."
        )

    def _init_weights(self) -> None:
        """Init weights of modules in head."""
        module_lists = [self.shared_fcs]
        module_lists += [self.fc_dep_uncer]
        module_lists += [self.fc_dep, self.dep_fcs]
        module_lists += [self.fc_dim, self.dim_fcs]
        module_lists += [self.fc_rot, self.rot_fcs]
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
                last_layer_dim *= np.prod(self.roi_pooler.resolution)
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

    def get_outputs(
        self,
        x_dep: torch.Tensor,
        x_dim: torch.Tensor,
        x_rot: torch.Tensor,
        x_2dc: torch.Tensor,
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
        delta_2dc = self.fc_2dc(x_2dc).view(-1, self.cls_out_channels, 2)
        return torch.cat([delta_2dc, depth, dim, alpha, depth_uncertainty], -1)

    def get_predictions(
        self,
        features_list: List[torch.Tensor],
        boxes: List[Boxes2D],
    ) -> List[torch.Tensor]:
        """Get 3D bounding box prediction parameters."""
        roi_feats = self.roi_pooler(features_list, boxes)
        x_dep, x_dim, x_rot, x_2dc = self.get_embeds(roi_feats)
        outputs: List[torch.Tensor] = self.get_outputs(
            x_dep, x_dim, x_rot, x_2dc
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

    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
        targets: LabelInstances,
    ) -> Tuple[LossesType, SamplingResult]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.
            targets: Targets corresponding to InputSamples.

        Returns:
            LossesType: A dict of scalar loss tensors.
            SamplingResult: Sampling result.
        """
        assert features is not None, "QD-3DT box3D head requires features!"
        features_list = [features[f] for f in self.in_features]

        # match and sample
        sampling_results = match_and_sample_proposals(
            self.matcher,
            self.sampler,
            boxes,
            inputs.targets.boxes2d,
            self.proposal_append_gt,
        )
        positives = [l == 1 for l in sampling_results.sampled_labels]
        pos_assigned_gt_inds = [
            i[p]
            for i, p in zip(sampling_results.sampled_target_indices, positives)
        ]
        pos_boxes = [
            b[p] for b, p in zip(sampling_results.sampled_boxes, positives)
        ]

        predictions = self.get_predictions(features_list, pos_boxes)

        tgt_params, labels = self.get_targets(
            pos_assigned_gt_inds, targets, inputs.intrinsics
        )
        loss = self.loss(
            torch.cat(predictions), torch.cat(tgt_params), torch.cat(labels)
        )
        return loss, sampling_results

    def forward_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
    ) -> List[Boxes3D]:
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
            dev = boxes[0].device
            return [Boxes3D.empty(dev) for _ in range(len(boxes))]

        features_list = [features[f] for f in self.in_features]
        predictions = self.get_predictions(features_list, boxes)

        return self.bbox_coder.decode(boxes, predictions, inputs.intrinsics)
