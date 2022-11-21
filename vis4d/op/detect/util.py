"""Detector utility functions."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from vis4d.op.box.encoder import BoxEncoder2D
from vis4d.op.box.matchers import Matcher
from vis4d.op.box.samplers import Sampler
from vis4d.op.util import unmap

from .anchor_generator import anchor_inside_image


class DetectorTargets(NamedTuple):
    """Targets for first-stage detection."""

    labels: torch.Tensor
    label_weights: torch.Tensor
    bbox_targets: torch.Tensor
    bbox_weights: torch.Tensor


def images_to_levels(targets) -> list[list[Tensor]]:
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
    box_encoder: BoxEncoder2D,
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
        box_encoder (BoxEncoder2D): Encodes boxes into target regression
            parameters.
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

    positives = sampling_result.sampled_labels == 1
    negatives = sampling_result.sampled_labels == 0
    pos_inds = sampling_result.sampled_box_indices[positives]
    pos_target_inds = sampling_result.sampled_target_indices[positives]
    neg_inds = sampling_result.sampled_box_indices[negatives]
    if len(pos_inds) > 0:
        pos_bbox_targets = box_encoder.encode(
            anchors[pos_inds],
            target_boxes[pos_target_inds],
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
