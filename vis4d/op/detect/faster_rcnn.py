"""Faster RCNN detector."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.box2d import apply_mask
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder
from vis4d.op.box.matchers import Matcher, MaxIoUMatcher
from vis4d.op.box.samplers import (
    RandomSampler,
    Sampler,
    match_and_sample_proposals,
)

from ..typing import Proposals, Targets
from .rcnn import RCNNHead, RCNNOut
from .rpn import RPN2RoI, RPNHead, RPNOut


class FRCNNOut(NamedTuple):
    """Faster RCNN function call outputs."""

    rpn: RPNOut
    roi: RCNNOut
    proposals: Proposals
    sampled_proposals: Proposals | None
    sampled_targets: Targets | None
    sampled_target_indices: list[torch.Tensor] | None


class FasterRCNNHead(nn.Module):
    """This class composes RPN and RCNN head components.

    It generates proposals via RPN and samples those, and runs the RCNN head
    on the sampled proposals. During training, the sampling process is based
    on the GT bounding boxes, during inference it is based on objectness score
    of the proposals.
    """

    def __init__(
        self,
        num_classes: int,
        anchor_generator: None | AnchorGenerator = None,
        rpn_box_decoder: None | DeltaXYWHBBoxDecoder = None,
        box_matcher: None | Matcher = None,
        box_sampler: None | Sampler = None,
        roi_head: None | RCNNHead = None,
        proposal_append_gt: bool = True,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of object categories.
            anchor_generator (AnchorGenerator, optional): Custom generator for
                RPN. Defaults to None.
            rpn_box_decoder (DeltaXYWHBBoxDecoder, optional): Custom rpn box
                decoder. Defaults to None.
            box_matcher (Matcher, optional): Custom box matcher for RCNN stage.
                Defaults to None.
            box_sampler (Sampler, optional): Custom box sampler for RCNN stage.
                Defaults to None.
            roi_head (RCNNHead, optional): Custom ROI head. Defaults to None.
            proposal_append_gt (bool): If to append the ground truth boxes for
                proposal sampling during training. Defaults to True.
        """
        super().__init__()
        if anchor_generator is None:
            anchor_generator = AnchorGenerator(
                scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
            )

        self.box_matcher = (
            MaxIoUMatcher(
                thresholds=[0.5],
                labels=[0, 1],
                allow_low_quality_matches=False,
            )
            if box_matcher is None
            else box_matcher
        )

        self.box_sampler = (
            RandomSampler(batch_size=512, positive_fraction=0.25)
            if box_sampler is None
            else box_sampler
        )

        self.proposal_append_gt = proposal_append_gt
        self.rpn_head = RPNHead(anchor_generator.num_base_priors[0])
        self.rpn2roi = RPN2RoI(anchor_generator, rpn_box_decoder)

        self.roi_head = (
            RCNNHead(num_classes=num_classes) if roi_head is None else roi_head
        )

    @torch.no_grad()
    def _sample_proposals(
        self,
        proposal_boxes: list[torch.Tensor],
        scores: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_classes: list[torch.Tensor],
    ) -> tuple[Proposals, Targets, list[torch.Tensor]]:
        """Sample proposals for training of Faster RCNN.

        Args:
            proposal_boxes (list[torch.Tensor]): Proposals decoded from RPN.
            scores (list[torch.Tensor]): Scores decoded from RPN.
            target_boxes (list[torch.Tensor]): All target boxes.
            target_classes (list[torch.Tensor]): According class labels.

        Returns:
            tuple[Proposals, Targets]: Sampled proposals, associated targets.
        """
        if self.proposal_append_gt:
            proposal_boxes = [
                torch.cat([p, t]) for p, t in zip(proposal_boxes, target_boxes)
            ]
            scores = [
                torch.cat([s, s.new_ones(len(t))])
                for s, t in zip(scores, target_boxes)
            ]

        (
            sampled_box_indices,
            sampled_target_indices,
            sampled_labels,
        ) = match_and_sample_proposals(
            self.box_matcher, self.box_sampler, proposal_boxes, target_boxes
        )

        sampled_boxes, sampled_scores = apply_mask(
            sampled_box_indices, proposal_boxes, scores
        )

        sampled_target_boxes, sampled_target_classes = apply_mask(
            sampled_target_indices, target_boxes, target_classes
        )

        sampled_proposals = Proposals(
            boxes=sampled_boxes, scores=sampled_scores
        )
        sampled_targets = Targets(
            boxes=sampled_target_boxes,
            classes=sampled_target_classes,
            labels=sampled_labels,
        )
        return sampled_proposals, sampled_targets, sampled_target_indices

    def forward(
        self,
        features: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
        target_boxes: None | list[torch.Tensor] = None,
        target_classes: None | list[torch.Tensor] = None,
    ) -> FRCNNOut:
        """Faster RCNN forward.

        Args:
            features (list[torch.Tensor]): Feature pyramid.
            images_hw (list[tuple[int, int]]): Image sizes without padding.
                This is necessary for removing the erroneous boxes on the
                padded regions.
            target_boxes (None | list[torch.Tensor], optional): Ground truth
                bounding box locations. Defaults to None.
            target_classes (None | list[torch.Tensor], optional): Ground truth
                bounding box classes. Defaults to None.

        Returns:
            FRCNNReturn: Proposal and RoI outputs.
        """
        if target_boxes is not None:
            assert target_classes is not None

        rpn_out = self.rpn_head(features)

        if target_boxes is not None:
            assert (
                target_classes is not None
            ), "Need target classes for target boxes!"
            proposal_boxes, scores = self.rpn2roi(
                rpn_out.cls, rpn_out.box, images_hw
            )

            (
                sampled_proposals,
                sampled_targets,
                sampled_target_indices,
            ) = self._sample_proposals(
                proposal_boxes, scores, target_boxes, target_classes
            )
            roi_out = self.roi_head(features, sampled_proposals.boxes)
        else:
            proposal_boxes, scores = self.rpn2roi(
                rpn_out.cls, rpn_out.box, images_hw
            )
            sampled_proposals, sampled_targets, sampled_target_indices = (
                None,
                None,
                None,
            )
            roi_out = self.roi_head(features, proposal_boxes)

        return FRCNNOut(
            roi=roi_out,
            rpn=rpn_out,
            proposals=Proposals(proposal_boxes, scores),
            sampled_proposals=sampled_proposals,
            sampled_targets=sampled_targets,
            sampled_target_indices=sampled_target_indices,
        )

    def __call__(
        self,
        features: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
        target_boxes: list[torch.Tensor] | None = None,
        target_classes: list[torch.Tensor] | None = None,
    ) -> FRCNNOut:
        """Type definition for call implementation."""
        return self._call_impl(
            features, images_hw, target_boxes, target_classes
        )
