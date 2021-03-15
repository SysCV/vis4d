"""RoI Heads definition for quasi-dense instance similarity learning"""
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.modeling.roi_heads import ROIHeads
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import ImageList, Instances

from openmt.core.build import build_matcher, build_sampler
from openmt.modeling.roi_heads import BaseRoIHead


class QDRoIHead(BaseRoIHead):
    """Instance embedding head for quasi-dense similarity learning."""

    def __init__(self, cfg):
        """Init."""
        self.sampler = build_sampler(cfg.sampler)
        self.matcher = build_matcher(cfg.matcher)

    def _init_embedding_head(cls, cfg, input_shape):
        """Init embedding head."""
        # TODO track head init
        pass

    def match_and_sample_proposals(self, batched_stuff):
        pass

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> List[
        Instances
    ]:  # TODO adapt similar design mmdetection, distinct train / inference forward functions
        """Forward of embedding head.
        We do not return a loss here, since the matching loss needs to
        be computed across all frames. Here, we only run the forward pass
        per frame, as well as proposal sampling and target assignment.
        """
        del images

        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.match_and_sample_proposals(proposals, targets)
        del targets

        pred_embeddings = self._forward_head(features, proposals)
        return pred_embeddings  # return optional association
