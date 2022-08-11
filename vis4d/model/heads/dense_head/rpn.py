"""Faster RCNN RPN Head."""
from math import prod
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms

from vis4d.common.bbox.anchor_generator import (
    AnchorGenerator,
    anchor_inside_flags,
)
from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxCoder
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import RandomSampler, SamplingResult
from vis4d.common.layers import Conv2d
from vis4d.model.losses.utils import smooth_l1_loss, weight_reduce_loss
from vis4d.struct import Boxes2D, LossesType


class RPNHead(nn.Module):
    """Faster RCNN RPN Head."""

    def __init__(
        self,
        num_convs: int = 1,
        in_channels: int = 256,
        feat_channels: int = 256,
        num_anchors: int = 3,
    ) -> None:
        """Init."""
        super().__init__()
        # TODO align num_anchors with anchor generator
        if num_convs > 1:
            rpn_convs = []
            for i in range(num_convs):
                if i > 0:
                    in_channels = feat_channels
                rpn_convs.append(
                    Conv2d(
                        in_channels,
                        feat_channels,
                        kernel_size=3,
                        padding=1,
                        activation=nn.ReLU(inplace=False),
                    )
                )
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = Conv2d(
                in_channels,
                feat_channels,
                kernel_size=3,
                padding=1,
                activation=nn.ReLU(inplace=True),
            )
        self.rpn_cls = Conv2d(feat_channels, num_anchors, 1)
        self.rpn_box = Conv2d(feat_channels, num_anchors * 4, 1)

    #     #  TODO weight init
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Conv2d):
    #         module.weight.data.normal_(mean=0.0, std=0.01)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],]:
        """Forward pass during training stage."""
        cls_outs, box_outs = [], []
        for feat in features:
            feat = self.rpn_conv(feat)
            cls_outs += [self.rpn_cls(feat)]
            box_outs += [self.rpn_box(feat)]
        return cls_outs, box_outs


class TransformRPNOutputs(nn.Module):
    def __init__(
        self,
        num_proposals_pre_nms: int = 2000,
        max_per_img: int = 1000,
        proposal_nms_threshold: float = 0.7,
        min_proposal_size: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.max_per_img = max_per_img
        self.min_proposal_size = min_proposal_size
        self.num_proposals_pre_nms = num_proposals_pre_nms
        self.proposal_nms_threshold = proposal_nms_threshold

        self.anchor_generator = AnchorGenerator(
            scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
        )

        self.bbox_coder = DeltaXYWHBBoxCoder(
            target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)
        )

    def _get_params_per_level(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        anchors: torch.Tensor,
    ):
        assert cls_out.size()[-2:] == reg_out.size()[-2:], (
            f"Shape mismatch: cls_out({cls_out.size()[-2:]}), reg_out("
            f"{reg_out.size()[-2:]})."
        )
        cls_out = cls_out.permute(1, 2, 0).reshape(-1).sigmoid()
        reg_out = reg_out.permute(1, 2, 0).reshape(-1, 4)
        if 0 < self.num_proposals_pre_nms < cls_out.shape[0]:
            cls_out_ranked, rank_inds = cls_out.sort(descending=True)
            topk_inds = rank_inds[: self.num_proposals_pre_nms]
            cls_out = cls_out_ranked[: self.num_proposals_pre_nms]
            reg_out = reg_out[topk_inds, :]
            anchors = anchors[topk_inds, :]

        return cls_out, reg_out, anchors

    def _decode_multi_level_outputs(
        self, cls_out_all, reg_out_all, anchors_all, level_all
    ) -> Boxes2D:
        scores = torch.cat(cls_out_all)
        levels = torch.cat(level_all)
        proposals = self.bbox_coder.decode(
            torch.cat(anchors_all),
            torch.cat(reg_out_all),
            max_shape=(512, 512),
        )  # TODO replace max_shape

        from vis4d.struct.labels.boxes import filter_boxes

        proposals, mask = filter_boxes(
            proposals, min_area=prod(self.min_proposal_size)
        )  # TODO area doesnt constrain size
        scores = scores[mask]
        levels = levels[mask]

        if proposals.numel() > 0:
            keep = batched_nms(
                proposals,
                scores,
                levels,
                iou_threshold=self.proposal_nms_threshold,
            )[: self.max_per_img]
            proposals = proposals[keep]
            scores = scores[keep]
        else:
            return Boxes2D(proposals.new_zeros(0, 5))
        return Boxes2D(torch.cat([proposals, scores.unsqueeze(-1)], -1))

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """

        Args:
            class_outs (N, 1 * A, H, W)
        Returns:
            boxes
            scores
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        device = class_outs[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        proposals_all = []
        for img_id in range(images_shape[0]):
            cls_out_all, reg_out_all, anchors_all, level_all = [], [], [], []
            for level in range(len(class_outs)):
                cls_out, reg_out, anchors = self._get_params_per_level(
                    class_outs[level][img_id],
                    regression_outs[level][img_id],
                    anchor_grids[level],
                )
                cls_out_all += [cls_out]
                reg_out_all += [reg_out]
                anchors_all += [anchors]
                level_all += [
                    cls_out.new_full((len(cls_out),), level, dtype=torch.long)
                ]

            proposals_all += [
                self._decode_multi_level_outputs(
                    cls_out_all, reg_out_all, anchors_all, level_all
                )
            ]
        return proposals_all


class RPNTargets(NamedTuple):
    labels: torch.Tensor
    label_weights: torch.Tensor
    bbox_targets: torch.Tensor
    bbox_weights: torch.Tensor


def unmap(data, count, inds, fill=0):  # TODO needed? if so revise
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


class RPNLoss(nn.Module):
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        bbox_coder: DeltaXYWHBBoxCoder,
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.bbox_coder = bbox_coder
        self.allowed_border = -1  # TODO should this be -1?
        self.matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        )
        self.sampler = RandomSampler(
            batch_size_per_image=256, positive_fraction=0.5
        )

    def _loss_single_scale(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        bbox_targets,
        bbox_weights,
        labels,
        label_weights,
        num_total_samples: int,
    ):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_out.permute(0, 2, 3, 1).reshape(-1)
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_score, labels, reduction="none"
        )
        loss_cls = weight_reduce_loss(
            loss_cls,
            label_weights,
            reduction="mean",
            avg_factor=num_total_samples,
        )
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = reg_out.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = smooth_l1_loss(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples
        )
        return loss_cls, loss_bbox

    def _get_targets_per_image(
        self,
        target_boxes: torch.Tensor,
        target_classes: torch.Tensor,
        anchors: torch.Tensor,
        valids: torch.Tensor,
    ) -> Tuple[RPNTargets, SamplingResult]:
        inside_flags = anchor_inside_flags(
            anchors,
            valids,
            (512, 512),  # TODO this is real im shape without padding
            allowed_border=self.allowed_border,
        )
        if not inside_flags.any():
            return RPNTargets()  # TODO implement
        # assign gt and sample anchors
        anchors = anchors[inside_flags, :]

        matching = self.matcher([Boxes2D(anchors)], [Boxes2D(target_boxes)])
        sampling_result = self.sampler(
            matching, [Boxes2D(anchors)], [Boxes2D(target_boxes)]
        )

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_zeros((num_valid_anchors,), dtype=torch.float)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        positives = sampling_result.sampled_labels[0] == 1
        negatives = sampling_result.sampled_labels[0] == 0
        pos_inds = sampling_result.sampled_indices[0][positives]
        neg_inds = sampling_result.sampled_indices[0][negatives]
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.sampled_boxes[0][positives].boxes,
                sampling_result.sampled_targets[0][positives].boxes,
            )
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = 1
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        num_total_anchors = anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (
            RPNTargets(labels, label_weights, bbox_targets, bbox_weights),
            sampling_result,
        )

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> LossesType:
        """RPN loss of faster rcnn.

        Args:
            outputs: Network outputs.
            targets (List[Boxes2D]): Target 2D boxes.
            metadata (Dict): Dictionary of metadata needed for loss, e.g.
                image size, feature map strides, etc.
        Returns:
            LossesType: Dictionary of scalar loss tensors.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = class_outs[0].device

        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        num_level_anchors = [anchors.size(0) for anchors in anchor_grids]
        valid_flags = self.anchor_generator.valid_flags(
            featmap_sizes, (512, 512), device  # TODO size
        )
        anchors_all_levels = torch.cat(anchor_grids)
        valids_all_levels = torch.cat(valid_flags)

        targets, num_total_pos, num_total_neg = [], 0, 0
        for tgt_box, tgt_cls in zip(target_boxes, target_classes):
            target, sampling_result = self._get_targets_per_image(
                tgt_box, tgt_cls, anchors_all_levels, valids_all_levels
            )
            num_total_pos += max(
                (sampling_result.sampled_labels[0] == 1).sum(), 1
            )
            num_total_neg += max(
                (sampling_result.sampled_labels[0] == 0).sum(), 1
            )
            bbox_targets_per_level = target.bbox_targets.split(
                num_level_anchors
            )
            bbox_weights_per_level = target.bbox_weights.split(
                num_level_anchors
            )
            labels_per_level = target.labels.split(num_level_anchors)
            label_weights_per_level = target.label_weights.split(
                num_level_anchors
            )
            targets.append(
                (
                    bbox_targets_per_level,
                    bbox_weights_per_level,
                    labels_per_level,
                    label_weights_per_level,
                )
            )
        targets_per_level = images_to_levels(targets)
        num_samples = num_total_pos + num_total_neg

        loss_cls_all, loss_bbox_all = torch.tensor(0.0), torch.tensor(0.0)
        for level_id, (cls_out, reg_out) in enumerate(
            zip(class_outs, regression_outs)
        ):
            loss_cls, loss_bbox = self._loss_single_scale(
                cls_out, reg_out, *targets_per_level[level_id], num_samples
            )
            loss_cls_all += loss_cls
            loss_bbox_all += loss_bbox
        return dict(rpn_loss_cls=loss_cls_all, rpn_loss_bbox=loss_bbox_all)


def images_to_levels(targets):
    """Convert targets by image to targets by feature level."""
    targets_per_level = []
    for lvl_id in range(len(targets[0][0])):
        targets_single_level = []
        for tgt_id in range(len(targets[0])):
            targets_single_level.append(
                torch.stack([tgt[tgt_id][lvl_id] for tgt in targets], 0)
            )
        targets_per_level.append(targets_single_level)
    return targets_per_level
