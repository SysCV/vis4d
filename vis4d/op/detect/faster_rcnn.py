"""Faster RCNN detector."""
from typing import List, NamedTuple, Optional

import torch
from torch import nn

from vis4d.common.bbox.anchor_generator import AnchorGenerator
from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxEncoder
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import (
    RandomSampler,
    match_and_sample_proposals,
)
from vis4d.op.heads.dense_head.rpn import TransformRPNOutputs

from ..heads.dense_head import RPNHead
from ..heads.roi_head.rcnn import RCNNHead


class FRCNNReturn(NamedTuple):
    rpn_cls_out: torch.Tensor
    rpn_reg_out: torch.Tensor
    # rpn: Optional[NamedTuple]  # TODO define
    roi_cls_out: torch.Tensor
    roi_reg_out: torch.Tensor
    proposal_boxes: List[torch.Tensor]
    proposal_scores: List[torch.Tensor]
    proposal_target_boxes: Optional[List[torch.Tensor]]
    proposal_target_classes: Optional[List[torch.Tensor]]
    proposal_labels: Optional[List[torch.Tensor]]


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


class FasterRCNN(nn.Module):
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

        self.rpn_head = RPNHead(self.anchor_generator.num_base_priors[0])
        self.rpn_head_transform = TransformRPNOutputs(
            self.anchor_generator, self.rpn_box_encoder
        )
        self.roi_head = RCNNHead(num_classes=num_classes)

    def forward(
        self,
        features: List[torch.Tensor],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> FRCNNReturn:
        """Faster RCNN forward.

        TODO(tobiasfshr) consider indiviual image sizes and paddings to
        remove invalid proposals.

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

        # TODO(tobiasfshr) RPN and RoI handle the whole feature pyramid
        rpn_cls_out, rpn_reg_out = self.rpn_head(features[2:])
        proposals, scores = self.rpn_head_transform(
            rpn_cls_out, rpn_reg_out, features[0].shape
        )

        if target_boxes is not None:
            (
                proposals,
                scores,
                sampled_target_boxes,
                sampled_target_classes,
                sampled_labels,
            ) = match_and_sample_proposals(
                self.box_matcher,
                self.box_sampler,
                proposals,
                scores,
                target_boxes,
                target_classes,
                proposal_append_gt=True,
            )

        else:
            sampled_target_boxes, sampled_target_classes, sampled_labels = (
                None,
                None,
                None,
            )

        roi_cls_out, roi_reg_out = self.roi_head(features[2:-1], proposals)

        return FRCNNReturn(
            rpn_cls_out=rpn_cls_out,
            rpn_reg_out=rpn_reg_out,
            roi_reg_out=roi_reg_out,
            roi_cls_out=roi_cls_out,
            proposal_boxes=proposals,
            proposal_scores=scores,
            proposal_target_boxes=sampled_target_boxes,
            proposal_target_classes=sampled_target_classes,
            proposal_labels=sampled_labels,
        )

    def __call__(
        self,
        features: torch.Tensor,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> FRCNNReturn:
        """Type definition for call implementation."""
        return self._call_impl(features, target_boxes, target_classes)
