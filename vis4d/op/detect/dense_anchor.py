"""Dense anchor-based head."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vis4d.common import TorchLossFunc
from vis4d.op.box.anchor import AnchorGenerator, anchor_inside_image
from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder
from vis4d.op.box.matchers import Matcher
from vis4d.op.box.samplers import Sampler
from vis4d.op.loss.reducer import SumWeightedLoss
from vis4d.op.util import unmap


class DetectorTargets(NamedTuple):
    """Targets for first-stage detection."""

    labels: Tensor
    label_weights: Tensor
    bbox_targets: Tensor
    bbox_weights: Tensor


def images_to_levels(
    targets: list[
        tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]
    ],
) -> list[list[Tensor]]:
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


def get_targets_per_image(
    target_boxes: Tensor,
    anchors: Tensor,
    matcher: Matcher,
    sampler: Sampler,
    box_encoder: DeltaXYWHBBoxEncoder,
    image_hw: tuple[int, int],
    target_class: Tensor | float = 1.0,
    allowed_border: int = 0,
) -> tuple[DetectorTargets, int, int]:
    """Get targets per batch element, all scales.

    Args:
        target_boxes (Tensor): (N, 4) Tensor of target boxes for a single
            image.
        anchors (Tensor): (M, 4) box priors
        matcher (Matcher): box matcher matching anchors to targets.
        sampler (Sampler): box sampler sub-sampling matches.
        box_encoder (DeltaXYWHBBoxEncoder): Encodes boxes into target
            regression parameters.
        image_hw (tuple[int, int]): input image height and width.
        target_class (Tensor | float, optional): class label(s) of target
            boxes. Defaults to 1.0.
        allowed_border (int, optional): Allowed border for sub-sampling anchors
            that lie inside the input image. Defaults to 0.

    Returns:
        tuple[DetectorTargets, Tensor, Tensor]: Targets, sum of positives, sum
            of negatives.
    """
    inside_flags = anchor_inside_image(
        anchors, image_hw, allowed_border=allowed_border
    )
    # assign gt and sample anchors
    anchors = anchors[inside_flags, :]

    matching = matcher(anchors, target_boxes)
    sampling_result = sampler(matching)

    num_valid_anchors = anchors.size(0)
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros((num_valid_anchors,))
    label_weights = anchors.new_zeros(num_valid_anchors)

    positives = torch.eq(sampling_result.sampled_labels, 1)
    negatives = torch.eq(sampling_result.sampled_labels, 0)
    pos_inds = sampling_result.sampled_box_indices[positives]
    pos_target_inds = sampling_result.sampled_target_indices[positives]
    neg_inds = sampling_result.sampled_box_indices[negatives]
    if len(pos_inds) > 0:
        pos_bbox_targets = box_encoder(
            anchors[pos_inds], target_boxes[pos_target_inds]
        )
        bbox_targets[pos_inds] = pos_bbox_targets
        bbox_weights[pos_inds] = 1.0
        if isinstance(target_class, float):
            labels[pos_inds] = target_class
        else:
            labels[pos_inds] = target_class[pos_target_inds].float()
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
        DetectorTargets(labels, label_weights, bbox_targets, bbox_weights),
        int(positives.sum()),
        int(negatives.sum()),
    )


def get_targets_per_batch(
    featmap_sizes: list[tuple[int, int]],
    target_boxes: list[Tensor],
    target_class_ids: list[Tensor | float],
    images_hw: list[tuple[int, int]],
    anchor_generator: AnchorGenerator,
    box_encoder: DeltaXYWHBBoxEncoder,
    box_matcher: Matcher,
    box_sampler: Sampler,
    allowed_border: int = 0,
) -> tuple[list[list[Tensor]], int]:
    """Get targets for all batch elements, all scales."""
    device = target_boxes[0].device

    anchor_grids = anchor_generator.grid_priors(featmap_sizes, device=device)
    num_level_anchors = [anchors.size(0) for anchors in anchor_grids]
    anchors_all_levels = torch.cat(anchor_grids)

    targets: list[
        tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]
    ] = []
    num_total_pos, num_total_neg = 0, 0
    for tgt_box, tgt_cls, image_hw in zip(
        target_boxes, target_class_ids, images_hw
    ):
        target, num_pos, num_neg = get_targets_per_image(
            tgt_box,
            anchors_all_levels,
            box_matcher,
            box_sampler,
            box_encoder,
            image_hw,
            tgt_cls,
            allowed_border,
        )
        num_total_pos += num_pos
        num_total_neg += num_neg
        bbox_targets_per_level = target.bbox_targets.split(num_level_anchors)
        bbox_weights_per_level = target.bbox_weights.split(num_level_anchors)
        labels_per_level = target.labels.split(num_level_anchors)
        label_weights_per_level = target.label_weights.split(num_level_anchors)
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
    return targets_per_level, num_samples


class DenseAnchorHeadLosses(NamedTuple):
    """Dense anchor head loss container."""

    loss_cls: Tensor
    loss_bbox: Tensor


class DenseAnchorHeadLoss(nn.Module):
    """Loss of dense anchor heads.

    For a given set of multi-scale dense outputs, compute the desired target
    outputs and apply classification and regression losses.
    The targets are computed with the given target bounding boxes, the
    anchor grid defined by the anchor generator and the given box encoder.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        box_matcher: Matcher,
        box_sampler: Sampler,
        loss_cls: TorchLossFunc,
        loss_bbox: TorchLossFunc,
        allowed_border: int = 0,
    ) -> None:
        """Creates an instance of the class.

        Args:
            anchor_generator (AnchorGenerator): Generates anchor grid priors.
            box_encoder (DeltaXYWHBBoxEncoder): Encodes bounding boxes to
                the desired network output.
            box_matcher (Matcher): Box matcher.
            box_sampler (Sampler): Box sampler.
            loss_cls (TorchLossFunc): Classification loss.
            loss_bbox (TorchLossFunc): Bounding box regression loss.
            allowed_border (int): The border to allow the valid anchor.
                Defaults to 0.
        """
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_encoder = box_encoder
        self.allowed_border = allowed_border
        self.matcher = box_matcher
        self.sampler = box_sampler
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

    def _loss_single_scale(
        self,
        cls_out: Tensor,
        reg_out: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        num_total_samples: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute losses per scale, all batch elements.

        Args:
            cls_out (Tensor): [N, C, H, W] tensor of class logits.
            reg_out (Tensor): [N, C, H, W] tensor of regression params.
            bbox_targets (Tensor): [H * W, 4] bounding box targets
            bbox_weights (Tensor): [H * W] per-sample weighting for loss.
            labels (Tensor): [H * W] classification targets.
            label_weights (Tensor): [H * W] per-sample weighting for loss.
            num_total_samples (int): average factor of loss.

        Returns:
            tuple[Tensor, Tensor]: classification and regression losses.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_out.permute(0, 2, 3, 1).reshape(labels.size(0), -1)
        if cls_score.size(1) > 1:
            labels = F.one_hot(  # pylint: disable=not-callable
                labels.long(), num_classes=cls_score.size(1) + 1
            )[:, : cls_score.size(1)].float()
            label_weights = label_weights.repeat(cls_score.size(1)).reshape(
                -1, cls_score.size(1)
            )
        else:
            cls_score = cls_score.squeeze(1)

        loss_cls = self.loss_cls(cls_score, labels, reduction="none")
        loss_cls = SumWeightedLoss(label_weights, num_total_samples)(loss_cls)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = reg_out.permute(0, 2, 3, 1).reshape(-1, 4)

        loss_bbox = self.loss_bbox(
            pred=bbox_pred,
            target=bbox_targets,
            reducer=SumWeightedLoss(bbox_weights, num_total_samples),
        )
        return loss_cls, loss_bbox

    def forward(
        self,
        cls_outs: list[Tensor],
        reg_outs: list[Tensor],
        target_boxes: list[Tensor],
        images_hw: list[tuple[int, int]],
        target_class_ids: list[Tensor | float] | None = None,
    ) -> DenseAnchorHeadLosses:
        """Compute RetinaNet classification and regression losses.

        Args:
            cls_outs (list[Tensor]): Network classification outputs
                at all scales.
            reg_outs (list[Tensor]): Network regression outputs
                at all scales.
            target_boxes (list[Tensor]): Target bounding boxes.
            images_hw (list[tuple[int, int]]): Image dimensions without
                padding.
            target_class_ids (list[Tensor] | None, optional): Target
                class labels.

        Returns:
            DenseAnchorHeadLosses: Classification and regression losses.
        """
        featmap_sizes = [
            (featmap.size()[-2], featmap.size()[-1]) for featmap in cls_outs
        ]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        if target_class_ids is None:
            target_class_ids = [1.0 for _ in range(len(target_boxes))]

        targets_per_level, num_samples = get_targets_per_batch(
            featmap_sizes,
            target_boxes,
            target_class_ids,
            images_hw,
            self.anchor_generator,
            self.box_encoder,
            self.matcher,
            self.sampler,
            self.allowed_border,
        )

        device = cls_outs[0].device
        loss_cls_all = torch.tensor(0.0, device=device)
        loss_bbox_all = torch.tensor(0.0, device=device)
        for level_id, (cls_out, reg_out) in enumerate(zip(cls_outs, reg_outs)):
            box_tgt, box_wgt, lbl, lbl_wgt = targets_per_level[level_id]
            loss_cls, loss_bbox = self._loss_single_scale(
                cls_out, reg_out, box_tgt, box_wgt, lbl, lbl_wgt, num_samples
            )
            loss_cls_all += loss_cls
            loss_bbox_all += loss_bbox
        return DenseAnchorHeadLosses(
            loss_cls=loss_cls_all, loss_bbox=loss_bbox_all
        )

    def __call__(
        self,
        cls_outs: list[Tensor],
        reg_outs: list[Tensor],
        target_boxes: list[Tensor],
        images_hw: list[tuple[int, int]],
        target_class_ids: list[Tensor] | None = None,
    ) -> DenseAnchorHeadLosses:
        """Type definition."""
        return self._call_impl(
            cls_outs, reg_outs, target_boxes, images_hw, target_class_ids
        )
