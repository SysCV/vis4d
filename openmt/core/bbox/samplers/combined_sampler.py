"""Combined Sampler."""
from typing import List, Optional, Tuple

import torch
from pydantic import validator

from openmt.structures import Boxes2D

from ..matchers.base_matcher import MatchResult
from ..utils import non_intersection, random_choice
from .base_sampler import BaseSampler, SamplerConfig


class CombinedSamplerConfig(SamplerConfig):
    batch_size_per_image: int
    positive_fraction: float
    pos_strategy: str
    neg_strategy: str

    # iou balanced strategy specific arguments
    floor_thr: Optional[float] = -1.0,
    floor_fraction: Optional[float] = 0.0,
    num_bins: int = 3

    @validator("pos_strategy", check_fields=False)
    def validate_pos_strategy(cls, v):
        if not v in ['instance_balanced', 'iou_balanced']:
            raise ValueError("pos_strategy must be in [instance_balanced, iou_balanced]")
        return v

    @validator("neg_strategy", check_fields=False)
    def validate_neg_strategy(cls, v):
        if not v in ['instance_balanced', 'iou_balanced']:
            raise ValueError("neg_strategy must be in [instance_balanced, iou_balanced]")
        return v


class CombinedSampler(BaseSampler):
    """Combined sampler. Can have different strategies for pos / neg samples"""

    def __init__(self, cfg: SamplerConfig):
        """Init."""
        super().__init__()
        self.cfg = CombinedSamplerConfig(**cfg.__dict__)
        self.bg_label = 0
        self.cfg.floor_thr = self.cfg.floor_thr[0]  # TODO fix
        self.cfg.floor_fraction = self.cfg.floor_fraction[0]  # TODO fix

    def instance_balanced_sampling(self, idx_tensor: torch.Tensor, assigned_gts: torch.Tensor,
                                   assigned_gt_ious: torch.Tensor,
                                   sample_size:
    int) \
            -> torch.Tensor:
        """Sample indices with balancing according to matched GT instance."""
        if idx_tensor.numel() <= sample_size:
            return idx_tensor
        else:
            unique_gt_inds = assigned_gts.unique()
            num_gts = len(unique_gt_inds)
            num_per_gt = int(round(sample_size / float(num_gts)) + 1)
            sampled_inds = []
            # sample specific amount per gt instance
            for i in unique_gt_inds:
                inds = torch.nonzero(assigned_gts == i, as_tuple=False)
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                else:
                    continue
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
            elif len(sampled_inds) > sample_size:
                sampled_inds = random_choice(sampled_inds, sample_size)
            return sampled_inds

    def iou_balanced_sampling(self, idx_tensor: torch.Tensor, assigned_gts: torch.Tensor, assigned_gt_ious: torch.Tensor,
                              sample_size: int) -> torch.Tensor:
        """Sample indices with balancing according to IoU with matched GT."""
        if idx_tensor.numel() <= sample_size:
            return idx_tensor
        else:
            # define 'floor' set - set with low iou samples
            if self.cfg.floor_thr >= 0:
                floor_set = idx_tensor[assigned_gt_ious <= self.cfg.floor_thr]
                iou_sampling_set = idx_tensor[assigned_gt_ious > self.cfg.floor_thr]
            else:
                floor_set = None
                iou_sampling_set = idx_tensor[assigned_gt_ious > self.cfg.floor_thr]

            num_iou_set_samples = int(sample_size * (1 - self.cfg.floor_fraction))
            if len(iou_sampling_set) > num_iou_set_samples:
                if self.cfg.num_bins >= 2:
                    iou_sampled_inds = self.sample_within_intervals(idx_tensor,
                        assigned_gt_ious, num_iou_set_samples)
                else:
                    iou_sampled_inds = random_choice(iou_sampling_set, num_iou_set_samples)
            else:
                iou_sampled_inds = iou_sampling_set

            if floor_set is not None:
                num_floor_set_samples = sample_size - len(iou_sampled_inds)
                if len(floor_set) > num_floor_set_samples:
                    sampled_floor_inds = random_choice(
                        floor_set, num_floor_set_samples)
                else:
                    sampled_floor_inds = floor_set
                sampled_inds = torch.cat(
                    [sampled_floor_inds, iou_sampled_inds])
            else:
                sampled_inds = iou_sampled_inds

            if len(sampled_inds) < sample_size:
                num_extra = sample_size - len(sampled_inds)
                extra_inds = non_intersection(idx_tensor, sampled_inds)
                if len(extra_inds) > num_extra:
                    extra_inds = random_choice(extra_inds, num_extra)
                sampled_inds = torch.cat([sampled_inds, extra_inds])

            return sampled_inds

    def sample(
        self,
        matching: MatchResult,
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], List[Boxes2D]]:
        """Sample boxes according to strategies defined in cfg."""
        pos_sample_size = int(self.cfg.batch_size_per_image * \
                          self.cfg.positive_fraction)
        sampled_boxes, sampled_targets = [], []
        for match, box, target in zip(matching, boxes, targets):
            positive_mask = (match.assigned_labels != -1) & (
                    match.assigned_labels != self.bg_label)
            negative_mask = match.assigned_labels == self.bg_label

            positive = nonzero_tuple(positive_mask)[0]
            negative = nonzero_tuple(negative_mask)[0]

            num_pos = min(positive.numel(), pos_sample_size)
            num_neg = self.cfg.batch_size_per_image - num_pos

            pos_idx = getattr(self, self.cfg.pos_strategy + '_sampling')(
                positive, match.assigned_gt_indices[positive_mask],
                match.assigned_gt_iou[positive_mask], num_pos)

            neg_idx = getattr(self, self.cfg.neg_strategy + '_sampling')(
                negative, match.assigned_gt_indices[negative_mask],
                match.assigned_gt_iou[negative_mask],  num_neg)

            sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)

            sampled_boxes.append(box[sampled_idxs])
            sampled_targets.append(
                target[match.assigned_gt_indices[sampled_idxs]]
            )

        return sampled_boxes, sampled_targets

    def sample_within_intervals(self, idx_tensor: torch.Tensor,
                                assigned_gt_ious: torch.Tensor,
                                sample_size: int):
        """Sample according to N iou intervals where N = num bins."""
        max_iou = assigned_gt_ious.max()
        iou_interval = (max_iou - self.cfg.floor_thr) / self.cfg.num_bins
        per_bin_samples = int(sample_size / self.cfg.num_bins)

        sampled_inds = []
        for i in range(self.cfg.num_bins):
            start_iou = self.cfg.floor_thr + i * iou_interval
            end_iou = self.cfg.floor_thr + (i + 1) * iou_interval
            tmp_set = ((start_iou <= assigned_gt_ious) & (assigned_gt_ious <
                       end_iou)).nonzero()[0]  # TODO fix
            if len(tmp_set) > per_bin_samples:
                tmp_sampled_set = random_choice(idx_tensor[tmp_set],
                                                     per_bin_samples)
            else:
                tmp_sampled_set = idx_tensor[tmp_set]
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = torch.cat(sampled_inds)
        if len(sampled_inds) < sample_size:
            num_extra = sample_size - len(sampled_inds)
            extra_inds = non_intersection(idx_tensor, sampled_inds)
            if len(extra_inds) > num_extra:
                extra_inds = random_choice(extra_inds, num_extra)
            sampled_inds = torch.cat([sampled_inds, extra_inds])

        return sampled_inds


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)
