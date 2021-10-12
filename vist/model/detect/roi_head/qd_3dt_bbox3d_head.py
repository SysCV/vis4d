"""3D Box Head definition for QD-3DT."""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from vist.common.bbox.coders import BaseBoxCoderConfig, build_box3d_coder
from vist.common.bbox.matchers import MatcherConfig, build_matcher
from vist.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from vist.common.bbox.samplers import (
    SamplerConfig,
    SamplingResult,
    build_sampler,
    match_and_sample_proposals,
)
from vist.common.geometry.rotation import generate_rotation_output
from vist.common.layers import add_conv_branch
from vist.model.losses import LossConfig, build_loss
from vist.struct import (
    Boxes2D,
    Boxes3D,
    InputSample,
    Intrinsics,
    LabelInstance,
    LossesType,
)

from .base import BaseRoIHead, BaseRoIHeadConfig


class QD3DTBBox3DHeadConfig(BaseRoIHeadConfig):
    """QD-3DT 3D Bounding Box Head config."""

    num_shared_convs: int = 2
    num_shared_fcs: int = 0
    num_dep_convs: int = 4
    num_dep_fcs: int = 0
    num_dim_convs: int = 4
    num_dim_fcs: int = 0
    num_rot_convs: int = 4
    num_rot_fcs: int = 0
    num_2dc_convs: int = 4
    num_2dc_fcs: int = 0
    in_channels: int = 256
    conv_out_dim: int = 256
    fc_out_dim: int = 1024
    roi_feat_size: int = 7
    num_classes: int
    conv_has_bias: bool = False
    norm: str
    num_groups: int = 32
    num_rotation_bins: int = 2

    loss: LossConfig
    box3d_coder: BaseBoxCoderConfig

    proposal_append_gt: bool
    in_features: List[str] = ["p2", "p3", "p4", "p5"]
    proposal_pooler: RoIPoolerConfig
    proposal_sampler: SamplerConfig
    proposal_matcher: MatcherConfig


class QD3DTBBox3DHead(BaseRoIHead):
    """QD-3DT 3D Bounding Box Head."""

    def __init__(self, cfg: BaseRoIHeadConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QD3DTBBox3DHeadConfig(**cfg.dict())

        self.cls_out_channels = self.cfg.num_classes

        self.sampler = build_sampler(self.cfg.proposal_sampler)
        self.matcher = build_matcher(self.cfg.proposal_matcher)
        self.roi_pooler = build_roi_pooler(self.cfg.proposal_pooler)

        self.bbox_coder = build_box3d_coder(self.cfg.box3d_coder)

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
        out_dim_dep = self.cls_out_channels
        self.fc_dep = nn.Linear(self.dep_last_dim, out_dim_dep)

        self.fc_dep_uncer = nn.Linear(self.dep_last_dim, out_dim_dep)

        out_dim_size = 3 * self.cls_out_channels
        self.fc_dim = nn.Linear(self.dim_last_dim, out_dim_size)

        out_rot_size = 3 * self.cfg.num_rotation_bins * self.cls_out_channels
        self.fc_rot = nn.Linear(self.rot_last_dim, out_rot_size)

        out_2dc_size = 2 * self.cls_out_channels
        self.fc_2dc = nn.Linear(self.cen_2d_last_dim, out_2dc_size)

        self._init_weights()

        # losses
        self.loss = build_loss(self.cfg.loss)
        assert (
            self.bbox_coder.cfg.num_rotation_bins  # type: ignore
            == self.cfg.num_rotation_bins
            == self.loss.cfg.num_rotation_bins
        ), (
            "num_rotation_bins must be consistent between head, "
            "loss and box coder."
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

    def get_outputs(
        self,
        x_dep: torch.Tensor,
        x_dim: torch.Tensor,
        x_rot: torch.Tensor,
        x_2dc: torch.Tensor,
    ) -> torch.Tensor:
        """Generate output 3D bounding box parameters."""
        depth = self.fc_dep(x_dep).view(-1, self.cfg.num_classes, 1)
        depth_uncertainty = self.fc_dep_uncer(x_dep).view(
            -1, self.cfg.num_classes, 1
        )
        dim = self.fc_dim(x_dim).view(-1, self.cfg.num_classes, 3)
        alpha = generate_rotation_output(
            self.fc_rot(x_rot), self.cfg.num_rotation_bins
        )
        delta_2dc = self.fc_2dc(x_2dc).view(-1, self.cfg.num_classes, 2)
        return torch.cat([delta_2dc, depth, dim, alpha, depth_uncertainty], -1)

    def forward(
        self,
        features_list: List[torch.Tensor],
        boxes: List[Boxes2D],
    ) -> List[torch.Tensor]:
        """Forward 3D bounding box estimation."""
        roi_feats = self.roi_pooler.pool(features_list, boxes)
        x_dep, x_dim, x_rot, x_2dc = self.get_embeds(roi_feats)
        outputs: List[torch.Tensor] = self.get_outputs(
            x_dep, x_dim, x_rot, x_2dc
        ).split([len(b) for b in boxes])
        return outputs

    def get_targets(
        self,
        pos_assigned_gt_inds: List[torch.Tensor],
        targets_2d: Sequence[Boxes2D],
        targets_3d: Sequence[Boxes3D],
        cam_intrinsics: Intrinsics,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get 3D bounding box targets for training."""
        targets_2d = [b[p] for b, p in zip(targets_2d, pos_assigned_gt_inds)]
        targets_3d = [b[p] for b, p in zip(targets_3d, pos_assigned_gt_inds)]

        bbox_targets = self.bbox_coder.encode(
            targets_2d, targets_3d, cam_intrinsics
        )

        labels = [
            t.class_ids[p] for t, p in zip(targets_2d, pos_assigned_gt_inds)
        ]
        return bbox_targets, labels

    def forward_train(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: Optional[Dict[str, torch.Tensor]] = None,
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
        assert features is not None, "QD-3DT box3D head requires features!"
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
        pos_boxes = [
            b[p] for b, p in zip(sampling_results.sampled_boxes, positives)
        ]

        predictions = self.forward(features_list, pos_boxes)

        targets, labels = self.get_targets(
            pos_assigned_gt_inds,
            inputs.boxes2d,
            inputs.boxes3d,
            inputs.intrinsics,
        )
        loss = self.loss(
            torch.cat(predictions), torch.cat(targets), torch.cat(labels)
        )
        return loss, sampling_results

    def forward_test(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Sequence[LabelInstance]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            List[LabelInstance]: Prediction output.
        """
        assert features is not None, "QD-3DT box3D head requires features!"
        if sum(len(b) for b in boxes) == 0:
            dev = boxes[0].device
            return [
                Boxes3D(
                    torch.empty(0, 8, device=dev),
                    torch.empty(0, device=dev),
                    torch.empty(0, device=dev),
                )
                for _ in range(len(boxes))
            ]

        features_list = [features[f] for f in self.cfg.in_features]
        predictions = self.forward(features_list, boxes)

        return self.bbox_coder.decode(boxes, predictions, inputs.intrinsics)
