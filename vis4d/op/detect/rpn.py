"""Faster RCNN RPN Head."""
from math import prod
from typing import List, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms

from vis4d.common.bbox.anchor_generator import (
    AnchorGenerator,
    anchor_inside_image,
)
from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxEncoder
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import RandomSampler, SamplingResult
from vis4d.common.layers import Conv2d
from vis4d.op.losses.utils import smooth_l1_loss, weight_reduce_loss
from vis4d.struct import Proposals
from vis4d.struct.labels.boxes import filter_boxes


class RPNOut(NamedTuple):
    """Output of RPN head."""

    # Sigmoid input for binary classification of the anchor
    # Positive means there is an object in that anchor.
    # Each list item is for on feature pyramid level.
    cls: List[torch.Tensor]
    # 4 x number of anchors for center offets and sizes (width, height) of the
    # boxes under the anchor.
    # Each list item is for on feature pyramid level.
    box: List[torch.Tensor]


class RPNHead(nn.Module):
    """Faster RCNN RPN Head.

    Creates RPN network output from a multi-scale feature map input.
    """

    rpn_conv: nn.Module

    def __init__(
        self,
        num_anchors: int,
        num_convs: int = 1,
        in_channels: int = 256,
        feat_channels: int = 256,
    ) -> None:
        """Init.

        Args:
            num_anchors (int): Number of anchors per cell.
            num_convs (int, optional): Number of conv layers before RPN heads. Defaults to 1.
            in_channels (int, optional): Feature channel size of input feature maps. Defaults to 256.
            feat_channels (int, optional): Feature channel size of conv layers. Defaults to 256.
        """
        super().__init__()
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Init RPN weights."""
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> RPNOut:
        """Forward pass of RPN."""
        cls_outs, box_outs = [], []
        for feat in features[2:]:  # Take stride 4 onwards
            feat = self.rpn_conv(feat)
            cls_outs += [self.rpn_cls(feat)]
            box_outs += [self.rpn_box(feat)]
        return RPNOut(cls=cls_outs, box=box_outs)

    def __call__(
        self,
        features: List[torch.Tensor],
    ) -> RPNOut:
        """Type definition"""
        return self._call_impl(features)


class RPN2RoI(nn.Module):
    """Generate Proposals (RoIs) from RPN network output.

    This class acts as a stateless functor that does the following:
    1. Create anchor grid for feature grids (classification and regression outputs) at all scales.
    For each image
        For each level
            2. Get a topk pre-selection of flattened classification scores and box energies from feature output before NMS.
        3. Decode class scores and box energies into proposal boxes, apply NMS.
    Return proposal boxes for all images.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        num_proposals_pre_nms: int = 2000,
        max_per_img: int = 1000,
        proposal_nms_threshold: float = 0.7,
        min_proposal_size: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_encoder = box_encoder
        self.max_per_img = max_per_img
        self.min_proposal_size = min_proposal_size
        self.num_proposals_pre_nms = num_proposals_pre_nms
        self.proposal_nms_threshold = proposal_nms_threshold

    def _get_params_per_level(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a topk pre-selection of flattened classification scores and box
        energies from feature output per level per image before nms.

        Args:
            cls_out (torch.Tensor): [C, H, W] classification scores at a particular scale.
            reg_out (torch.Tensor): [C, H, W] regression parameters at a particular scale.
            anchors (torch.Tensor): [H*W, 4] anchor boxes per cell.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: topk flattened
                classification, regression outputs and corresponding anchors.
        """
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
        self,
        cls_out_all: List[torch.Tensor],
        reg_out_all: List[torch.Tensor],
        anchors_all: List[torch.Tensor],
        level_all: List[torch.Tensor],
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode box energies into proposals for a single image, post-process
        via NMS. NMS is performed per level. Afterwards, select topk proposals.

        Args:
            cls_out_all (List[torch.Tensor]): topk class scores per level.
            reg_out_all (List[torch.Tensor]): topk regression params per level.
            anchors_all (List[torch.Tensor]): topk anchor boxes per level.
            level_all (List[torch.Tensor]): tensors indicating level per entry.
            image_hw (Tuple[int, int]): image size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: decoded proposal boxes & scores.
        """
        scores = torch.cat(cls_out_all)
        levels = torch.cat(level_all)
        proposals = self.box_encoder.decode(
            torch.cat(anchors_all),
            torch.cat(reg_out_all),
            max_shape=image_hw,
        )

        proposals, mask = filter_boxes(
            proposals, min_area=prod(self.min_proposal_size)
        )
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
            return proposals.new_zeros(0, 4), scores.new_zeros(
                0,
            )
        return proposals, scores

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
    ) -> Proposals:
        """Compute proposals from RPN network outputs.

        Generate anchor grid for all scales.
        For each batch element:
            Compute classification, regression and anchor pairs for all scales.
            Decode those pairs into proposals, post-process with NMS.

        Args:
            class_outs (List[torch.Tensor]): [N, 1 * A, H, W] per scale.
            regression_outs (List[torch.Tensor]): [N, 4 * A, H, W] per scale.
            images_hw (List[Tuple[int, int]]): list of image sizes.

        Returns:
            Proposals: proposal boxes and scores.
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        device = class_outs[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        proposals, scores = [], []
        for img_id, image_hw in enumerate(images_hw):
            cls_out_all, reg_out_all, anchors_all, level_all = [], [], [], []
            for level, (cls_outs, reg_outs, anchor_grid) in enumerate(
                zip(class_outs, regression_outs, anchor_grids)
            ):
                cls_out, reg_out, anchors = self._get_params_per_level(
                    cls_outs[img_id],
                    reg_outs[img_id],
                    anchor_grid,
                )
                cls_out_all += [cls_out]
                reg_out_all += [reg_out]
                anchors_all += [anchors]
                level_all += [
                    cls_out.new_full((len(cls_out),), level, dtype=torch.long)
                ]

            box, score = self._decode_multi_level_outputs(
                cls_out_all, reg_out_all, anchors_all, level_all, image_hw
            )
            proposals.append(box)
            scores.append(score)
        return Proposals(proposals, scores)


class RPNTargets(NamedTuple):
    """Targets for RPNLoss."""

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


class RPNLosses(NamedTuple):
    """RPN loss container."""

    rpn_loss_cls: torch.Tensor
    rpn_loss_bbox: torch.Tensor


class RPNLoss(nn.Module):
    """Loss of region proposal network.

    For a given set of multi-scale RPN outputs, compute the desired target
    outputs and apply classification and regression losses.
    The targets are computed with the given target bounding boxes, the
    anchor grid defined by the anchor generator and the given box encoder.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
    ):
        """Init.

        Args:
            anchor_generator (AnchorGenerator): Generates anchor grid priors.
            box_encoder (DeltaXYWHBBoxEncoder): Encodes bounding boxes to
                the desired network output.
        """
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_encoder = box_encoder
        self.allowed_border = 0
        self.matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        )
        self.sampler = RandomSampler(batch_size=256, positive_fraction=0.5)

    def _loss_single_scale(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_weights: torch.Tensor,
        labels: torch.Tensor,
        label_weights: torch.Tensor,
        num_total_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute losses per scale, all batch elements.

        Args:
            cls_out (torch.Tensor): [N, C, H, W] tensor of class logits.
            reg_out (torch.Tensor): [N, C, H, W] tensor of regression params.
            bbox_targets (torch.Tensor): [H*W, 4] bounding box targets
            bbox_weights (torch.Tensor): [H*W] per-sample weighting for loss.
            labels (torch.Tensor): [H*W] classification targets.
            label_weights (torch.Tensor): [H*W] per-sample weighting for loss.
            num_total_samples (int): average factor of loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: classification and regression
                losses.
        """
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
        anchors: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Tuple[RPNTargets, int, int]:
        """Get targets per batch element, all scales."""
        inside_flags = anchor_inside_image(
            anchors,
            image_hw,
            allowed_border=self.allowed_border,
        )
        # assign gt and sample anchors
        anchors = anchors[inside_flags, :]

        matching = self.matcher(anchors, target_boxes)
        sampling_result = self.sampler(matching)

        num_valid_anchors = anchors.size(0)
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_zeros((num_valid_anchors,))
        label_weights = anchors.new_zeros(num_valid_anchors)

        positives = sampling_result.sampled_labels == 1
        negatives = sampling_result.sampled_labels == 0
        pos_inds = sampling_result.sampled_box_indices[positives]
        pos_target_inds = sampling_result.sampled_target_indices[positives]
        neg_inds = sampling_result.sampled_box_indices[negatives]
        if len(pos_inds) > 0:
            pos_bbox_targets = self.box_encoder.encode(
                anchors[pos_inds],
                target_boxes[pos_target_inds],
            )
            bbox_targets[pos_inds] = pos_bbox_targets
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = 1.0
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        num_total_anchors = inside_flags.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (
            RPNTargets(labels, label_weights, bbox_targets, bbox_weights),
            positives.sum(),
            negatives.sum(),
        )

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
    ) -> RPNLosses:
        """Compute RPN classification and regression losses.

        Args:
            class_outs (List[torch.Tensor]): Network classification outputs at all scales.
            regression_outs (List[torch.Tensor]): Network regression outputs at all scales.
            target_boxes (List[torch.Tensor]): Target bounding boxes.
            images_hw (List[Tuple[int, int]]): Image dimensions without padding.

        Returns:
            RPNLosses: classification and regression losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = class_outs[0].device

        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        num_level_anchors = [anchors.size(0) for anchors in anchor_grids]
        anchors_all_levels = torch.cat(anchor_grids)

        targets, num_total_pos, num_total_neg = [], 0, 0
        for tgt_box, image_hw in zip(target_boxes, images_hw):
            target, num_pos, num_neg = self._get_targets_per_image(
                tgt_box,
                anchors_all_levels,
                image_hw,
            )
            num_total_pos += num_pos
            num_total_neg += num_neg
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

        loss_cls_all = torch.tensor(0.0, device=device)
        loss_bbox_all = torch.tensor(0.0, device=device)
        for level_id, (cls_out, reg_out) in enumerate(
            zip(class_outs, regression_outs)
        ):
            loss_cls, loss_bbox = self._loss_single_scale(
                cls_out, reg_out, *targets_per_level[level_id], num_samples
            )
            loss_cls_all += loss_cls
            loss_bbox_all += loss_bbox
        return RPNLosses(
            rpn_loss_cls=loss_cls_all, rpn_loss_bbox=loss_bbox_all
        )

    def __call__(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
    ) -> RPNLosses:
        """Type definition."""
        return self._call_impl(
            class_outs, regression_outs, target_boxes, images_hw
        )


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
