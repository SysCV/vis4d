"""Mask RCNN detector."""
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from vis4d.common.bbox.anchor_generator import AnchorGenerator
from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxEncoder
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import RandomSampler

from ..heads.roi_head.rcnn import MaskRCNNHead
from .faster_rcnn import FasterRCNN, FRCNNOut


class MaskRCNNReturn(NamedTuple):
    frcnn_out: FRCNNOut
    roi_mask_out: torch.Tensor
    proposal_masks: Optional[List[torch.Tensor]]


class MaskRCNN(nn.Module):
    """mmdetection mask rcnn wrapper."""

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
        self.faster_rcnn = FasterRCNN(
            num_classes,
            anchor_generator,
            rpn_box_encoder,
            rcnn_box_encoder,
            box_matcher,
            box_sampler,
        )
        self.mask_head = MaskRCNNHead(num_classes=num_classes)

    def forward(
        self,
        features: List[torch.Tensor],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        target_masks: Optional[List[torch.Tensor]] = None,
    ) -> MaskRCNNReturn:
        """_summary_
        # TODO(thomaseh)

        Args:
            features (List[torch.Tensor]): _description_
            target_boxes (Optional[List[torch.Tensor]], optional): _description_. Defaults to None.
            target_classes (Optional[List[torch.Tensor]], optional): _description_. Defaults to None.
            target_masks (Optional[List[torch.Tensor]], optional): _description_. Defaults to None.

        Returns:
            MaskRCNNReturn: _description_
        """
        rcnn_outs = self.faster_rcnn(features, target_boxes, target_classes)
        roi_mask_out = self.mask_head(
            features[2:-1], rcnn_outs.proposals.boxes
        )

        if target_boxes is not None and target_masks is not None:
            assert rcnn_outs.proposals.target_indices is not None
            sampled_target_masks: Optional[List[torch.Tensor]] = [
                tmask[tind]
                for tmask, tind in zip(
                    target_masks, rcnn_outs.proposals.target_indices
                )
            ]
        else:
            sampled_target_masks = None

        return MaskRCNNReturn(
            frcnn_out=rcnn_outs,
            roi_mask_out=roi_mask_out,
            proposal_masks=sampled_target_masks,
        )

    def __call__(
        self,
        images: torch.Tensor,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        target_masks: Optional[List[torch.Tensor]] = None,
    ) -> MaskRCNNReturn:
        """Type definition for call implementation."""
        return self._call_impl(
            images, target_boxes, target_classes, target_masks
        )
