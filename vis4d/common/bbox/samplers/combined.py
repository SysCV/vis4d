"""Combined Sampler."""
from collections import defaultdict
from typing import Dict, List, Union

import torch

from vis4d.struct import ArgsType

from ..matchers.base import MatchResult
from ..utils import non_intersection, random_choice
from .base import BaseSampler, SamplingResult
from .utils import add_to_result


class CombinedSampler(BaseSampler):
    """Combined sampler. Can have different strategies for pos/neg samples."""

    def __init__(
        self,
        *args: ArgsType,
        pos_strategy: str,
        neg_strategy: str,
        neg_pos_ub: float = 3.0,
        floor_thr: float = -1.0,
        floor_fraction: float = 0.0,
        num_bins: int = 3,
        bg_label: int = 0,
        **kwargs: ArgsType
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.neg_pos_ub = neg_pos_ub
        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins
        self.bg_label = bg_label

        if (
            not pos_strategy
            in [
                "instance_balanced",
                "iou_balanced",
            ]
            or not neg_strategy in ["instance_balanced", "iou_balanced"]
        ):
            raise ValueError(
                "strategies must be in [instance_balanced, iou_balanced]"
            )

        self.pos_strategy = getattr(self, pos_strategy + "_sampling")
        self.neg_strategy = getattr(self, neg_strategy + "_sampling")

    @staticmethod
    def instance_balanced_sampling(
        idx_tensor: torch.Tensor,
        assigned_gts: torch.Tensor,
        assigned_gt_ious: torch.Tensor,  # pylint: disable=unused-argument
        sample_size: int,
    ) -> torch.Tensor:
        """Sample indices with balancing according to matched GT instance."""
        if idx_tensor.numel() <= sample_size:
            return idx_tensor

        unique_gt_inds = assigned_gts.unique()
        num_gts = len(unique_gt_inds)
        num_per_gt = int(sample_size / float(num_gts))
        sampled_inds = []
        # sample specific amount per gt instance
        for i in unique_gt_inds:
            inds = torch.nonzero(assigned_gts == i, as_tuple=False)
            inds = inds.squeeze(1)
            if len(inds) > num_per_gt:
                inds = random_choice(inds, num_per_gt)
            sampled_inds.append(inds)
        sampled_inds = torch.cat(sampled_inds)

        # deal with edge cases
        if len(sampled_inds) < sample_size:
            num_extra = sample_size - len(sampled_inds)
            extra_inds = non_intersection(idx_tensor, sampled_inds)
            if len(extra_inds) > num_extra:
                extra_inds = random_choice(extra_inds, num_extra)
            sampled_inds = torch.cat([sampled_inds, extra_inds])
        return sampled_inds

    def iou_balanced_sampling(
        self,
        idx_tensor: torch.Tensor,
        assigned_gts: torch.Tensor,  # pylint: disable=unused-argument
        assigned_gt_ious: torch.Tensor,
        sample_size: int,
    ) -> torch.Tensor:
        """Sample indices with balancing according to IoU with matched GT."""
        if idx_tensor.numel() <= sample_size:
            return idx_tensor

        # define 'floor' set - set with low iou samples
        if self.floor_thr >= 0:
            floor_set = idx_tensor[assigned_gt_ious <= self.floor_thr]
            iou_sampling_set = idx_tensor[assigned_gt_ious > self.floor_thr]
        else:
            floor_set = None
            iou_sampling_set = idx_tensor[assigned_gt_ious > self.floor_thr]

        num_iou_set_samples = int(sample_size * (1 - self.floor_fraction))
        if len(iou_sampling_set) > num_iou_set_samples:
            if self.num_bins >= 2:
                iou_sampled_inds = self.sample_within_intervals(
                    idx_tensor, assigned_gt_ious, num_iou_set_samples
                )
            else:
                iou_sampled_inds = random_choice(
                    iou_sampling_set, num_iou_set_samples
                )
        else:
            iou_sampled_inds = iou_sampling_set  # pragma: no cover

        if floor_set is not None:
            num_floor_set_samples = sample_size - len(iou_sampled_inds)
            if len(floor_set) > num_floor_set_samples:
                sampled_floor_inds = random_choice(
                    floor_set, num_floor_set_samples
                )
            else:
                sampled_floor_inds = floor_set  # pragma: no cover
            sampled_inds = torch.cat([sampled_floor_inds, iou_sampled_inds])
        else:
            sampled_inds = iou_sampled_inds

        if len(sampled_inds) < sample_size:  # pragma: no cover
            num_extra = sample_size - len(sampled_inds)
            extra_inds = non_intersection(idx_tensor, sampled_inds)
            if len(extra_inds) > num_extra:
                extra_inds = random_choice(extra_inds, num_extra)
            sampled_inds = torch.cat([sampled_inds, extra_inds])

        return sampled_inds

    def forward(
        self,
        matching: MatchResult,
        boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> SamplingResult:
        """Sample boxes according to strategies defined in cfg."""
        pos_sample_size = int(
            self.batch_size_per_image * self.positive_fraction
        )

        positive_mask = (matching.assigned_labels != -1) & (
            matching.assigned_labels != self.bg_label
        )
        negative_mask = matching.assigned_labels == self.bg_label

        positive = positive_mask.nonzero()[:, 0]
        negative = negative_mask.nonzero()[:, 0]

        num_pos = min(positive.numel(), pos_sample_size)
        num_neg = self.batch_size_per_image - num_pos

        if self.neg_pos_ub >= 0:
            neg_upper_bound = int(self.neg_pos_ub * num_pos)
            num_neg = min(num_neg, neg_upper_bound)

        pos_idx = self.pos_strategy(
            idx_tensor=positive,
            assigned_gts=matching.assigned_gt_indices.long()[positive_mask],
            assigned_gt_ious=matching.assigned_gt_iou[positive_mask],
            sample_size=num_pos,
        )

        neg_idx = self.neg_strategy(
            idx_tensor=negative,
            assigned_gts=matching.assigned_gt_indices.long()[negative_mask],
            assigned_gt_ious=matching.assigned_gt_iou[negative_mask],
            sample_size=num_neg,
        )
        sampled_idcs = torch.cat([pos_idx, neg_idx], dim=0)

        return SamplingResult(
            sampled_boxes=boxes[sampled_idcs],
            sampled_target_boxes=target_boxes[
                matching.assigned_gt_indices.long()[sampled_idcs]
            ],  # TODO why .long(), should already be long
            sampled_target_classes=target_classes[
                matching.assigned_gt_indices.long()[sampled_idcs]
            ],
            sampled_labels=matching.assigned_labels[sampled_idcs],
            sampled_indices=sampled_idcs,
            sampled_target_indices=matching.assigned_gt_indices[sampled_idcs],
        )

    def sample_within_intervals(
        self,
        idx_tensor: torch.Tensor,
        assigned_gt_ious: torch.Tensor,
        sample_size: int,
    ) -> torch.Tensor:
        """Sample according to N iou intervals where N = num bins."""
        floor_thr = max(self.floor_thr, 0.0)
        max_iou = assigned_gt_ious.max()
        iou_interval = (max_iou - floor_thr) / self.num_bins
        per_bin_samples = int(sample_size / self.num_bins)

        sampled_inds = []
        for i in range(self.num_bins):
            start_iou = floor_thr + i * iou_interval
            end_iou = floor_thr + (i + 1) * iou_interval
            tmp_set = (
                (start_iou <= assigned_gt_ious) & (assigned_gt_ious < end_iou)
            ).nonzero()[:, 0]
            if len(tmp_set) > per_bin_samples:
                tmp_sampled_set = random_choice(
                    idx_tensor[tmp_set], per_bin_samples
                )
            else:
                tmp_sampled_set = idx_tensor[tmp_set]  # pragma: no cover
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = torch.cat(sampled_inds)
        if len(sampled_inds) < sample_size:
            num_extra = sample_size - len(sampled_inds)
            extra_inds = non_intersection(idx_tensor, sampled_inds)
            if len(extra_inds) > num_extra:
                extra_inds = random_choice(extra_inds, num_extra)
            sampled_inds = torch.cat([sampled_inds, extra_inds])

        return sampled_inds
