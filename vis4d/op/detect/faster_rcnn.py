"""Faster RCNN detector."""
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from vis4d.common.bbox.anchor_generator import AnchorGenerator
from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxEncoder
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import (
    RandomSampler,
    match_and_sample_proposals,
)
from vis4d.common.bbox.utils import apply_mask
from vis4d.op.heads.dense_head.rpn import TransformRPNOutputs
from vis4d.struct import Proposals

from ..heads.dense_head.rpn import RPNHead, RPNOut
from ..heads.roi_head.rcnn import RCNNHead, RCNNOut


class Targets(NamedTuple):
    """Output structure for targets."""

    boxes: List[torch.Tensor]
    classes: List[torch.Tensor]
    labels: List[torch.Tensor]


class FRCNNOut(NamedTuple):
    """Faster RCNN function call outputs."""

    rpn: RPNOut
    roi: RCNNOut
    proposals: Proposals
    sampled_proposals: Optional[Proposals]
    sampled_targets: Optional[Targets]
    sampled_target_indices: Optional[List[torch.Tensor]]


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
    )


def get_default_rpn_box_encoder() -> DeltaXYWHBBoxEncoder:
    """Get the default bounding box encoder for RPN."""
    return DeltaXYWHBBoxEncoder(
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
    )


def get_default_rcnn_box_encoder() -> DeltaXYWHBBoxEncoder:
    """Get the default bounding box encoder for RCNN."""
    return DeltaXYWHBBoxEncoder(
        clip_border=True,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
    )


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default bounding box matcher."""
    return MaxIoUMatcher(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    )


def get_default_box_sampler() -> RandomSampler:
    """Get default bounding box sampler."""
    return RandomSampler(batch_size=512, positive_fraction=0.25)


class FasterRCNNHead(nn.Module):
    """mmdetection two-stage detector wrapper."""

    def __init__(
        self,
        num_classes: int = 80,
        anchor_generator: Optional[AnchorGenerator] = None,
        rpn_box_encoder: Optional[DeltaXYWHBBoxEncoder] = None,
        rcnn_box_encoder: Optional[DeltaXYWHBBoxEncoder] = None,
        box_matcher: Optional[MaxIoUMatcher] = None,
        box_sampler: Optional[RandomSampler] = None,
    ):
        """Init."""
        super().__init__()
        self.anchor_generator = (
            anchor_generator
            if anchor_generator is not None
            else get_default_anchor_generator()
        )
        self.rpn_box_encoder = (
            rpn_box_encoder
            if rpn_box_encoder is not None
            else get_default_rpn_box_encoder()
        )
        self.rcnn_box_encoder = (
            rcnn_box_encoder
            if rcnn_box_encoder is not None
            else get_default_rcnn_box_encoder()
        )
        self.box_matcher = (
            box_matcher
            if box_matcher is not None
            else get_default_box_matcher()
        )
        self.box_sampler = (
            box_sampler
            if box_sampler is not None
            else get_default_box_sampler()
        )
        self.proposal_append_gt = True  # TODO make option
        self.rpn_head = RPNHead(self.anchor_generator.num_base_priors[0])
        self.rpn_head_transform = TransformRPNOutputs(
            self.anchor_generator, self.rpn_box_encoder
        )
        self.roi_head = RCNNHead(num_classes=num_classes)

    def _sample_proposals(
        self,
        proposal_boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> Tuple[Proposals, Targets, List[torch.Tensor]]:
        """Sample proposals for training of Faster RCNN.

        Args:
            proposal_boxes (List[torch.Tensor]): proposals decoded from RPN.
            scores (List[torch.Tensor]): scores decoded from RPN.
            target_boxes (List[torch.Tensor]): all target boxes.
            target_classes (List[torch.Tensor]): according class labels.

        Returns:
            Tuple[Proposals, Targets]: Sampled proposals, associated targets.
        """
        if self.proposal_append_gt:
            proposal_boxes = [
                torch.cat([p, t]) for p, t in zip(proposal_boxes, target_boxes)
            ]
            scores = [
                torch.cat(
                    [
                        s,
                        s.new_ones(
                            len(t),
                        ),
                    ]
                )
                for s, t in zip(scores, target_boxes)
            ]

        (
            sampled_box_indices,
            sampled_target_indices,
            sampled_labels,
        ) = match_and_sample_proposals(
            self.box_matcher,
            self.box_sampler,
            proposal_boxes,
            target_boxes,
        )

        sampled_boxes, sampled_scores = apply_mask(
            sampled_box_indices, proposal_boxes, scores
        )

        sampled_target_boxes, sampled_target_classes = apply_mask(
            sampled_target_indices, target_boxes, target_classes
        )

        sampled_proposals = Proposals(
            boxes=sampled_boxes,
            scores=sampled_scores,
        )
        sampled_targets = Targets(
            boxes=sampled_target_boxes,
            classes=sampled_target_classes,
            labels=sampled_labels,
        )
        return sampled_proposals, sampled_targets, sampled_target_indices

    def forward(
        self,
        features: List[torch.Tensor],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> FRCNNOut:
        """Faster RCNN forward.

        Args:
            features (List[torch.Tensor]): Feature pyramid
            target_boxes (Optional[List[torch.Tensor]], optional): Ground
            truth bounding box locations. Defaults to None.
            target_classes (Optional[List[torch.Tensor]], optional): Ground
            truth bounding box classes. Defaults to None.

        Returns:
            FRCNNReturn: proposal and roi outputs.
        """
        if target_boxes is not None:
            assert target_classes is not None

        rpn_out = self.rpn_head(features)
        image_hw = features[0].shape[2:]

        if target_boxes is not None:
            assert (
                target_classes is not None
            ), "Need target classes for target boxes!"

            self.rpn_head_transform.num_proposals_pre_nms = 2000
            proposal_boxes, scores = self.rpn_head_transform(
                rpn_out.cls, rpn_out.box, image_hw
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
            self.rpn_head_transform.num_proposals_pre_nms = 1000
            proposal_boxes, scores = self.rpn_head_transform(
                rpn_out.cls, rpn_out.box, image_hw
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
        features: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> FRCNNOut:
        """Type definition for call implementation."""
        return self._call_impl(
            features, images_hw, target_boxes, target_classes
        )
