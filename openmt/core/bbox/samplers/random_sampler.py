"""Random Sampler."""
from typing import List, Tuple

import torch
from detectron2.modeling.sampling import subsample_labels

from openmt.structures import Boxes2D

from ..matchers.base_matcher import MatchResult
from .base_sampler import BaseSampler, SamplerConfig


class RandomSamplerConfig(SamplerConfig):
    batch_size_per_image: int
    positive_fraction: float
    proposal_append_gt: bool


class RandomSampler(BaseSampler):
    """Random sampler class."""

    def __init__(self, cfg: SamplerConfig):
        """Init."""
        super().__init__()
        self.cfg = RandomSamplerConfig(**cfg.__dict__)

        self.batch_size_per_image = self.cfg.batch_size_per_image
        self.positive_fraction = self.cfg.positive_fraction
        # self.num_classes = cfg.num_classes  TODO num classes really needed?
        self.proposal_append_gt = self.cfg.proposal_append_gt
        self.bg_label = 0

    def sample(
        self,
        matching: MatchResult,
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], List[Boxes2D]]:
        """Sample boxes randomly."""

        sampled_boxes, sampled_targets = [], []
        for match, box, target in zip(matching, boxes, targets):
            pos_idx, neg_idx = subsample_labels(
                match.assigned_labels,
                self.batch_size_per_image,
                self.positive_fraction,
                self.bg_label,
            )
            sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)

            sampled_boxes.append(
                target[match.assigned_gt_indices[sampled_idxs]]
            )
            sampled_targets.append(box[sampled_idxs])

        return boxes, targets
