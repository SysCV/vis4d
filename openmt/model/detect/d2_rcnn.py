"""Detectron2 detector wrapper."""
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.modeling import GeneralizedRCNN

from openmt.model.detect.d2_utils import (
    D2GeneralizedRCNNConfig,
    detections_to_box2d,
    model_to_detectron2,
    proposal_to_box2d,
    target_to_instance,
)
from openmt.struct import Boxes2D, DetectionOutput, ImageList

from .base import BaseDetector, BaseDetectorConfig


class D2GeneralizedRCNN(BaseDetector):
    """Detectron2 detector wrapper."""

    def __init__(self, cfg: BaseDetectorConfig):
        """Init."""
        super().__init__()
        self.cfg = D2GeneralizedRCNNConfig(**cfg.dict())
        self.d2_cfg = model_to_detectron2(self.cfg)
        # pylint: disable=too-many-function-args,missing-kwoa
        self.d2_detector = GeneralizedRCNN(self.d2_cfg)

    @property
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        return self.d2_detector.pixel_mean.device

    def preprocess_image(
        self, batched_inputs: Tuple[Dict[str, torch.Tensor]]
    ) -> ImageList:
        """Normalize, pad and batch the input images."""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [
            (x - self.d2_detector.pixel_mean) / self.d2_detector.pixel_std
            for x in images
        ]
        images = ImageList.from_tensors(
            images, self.d2_detector.backbone.size_divisibility
        )
        return images

    def forward(
        self,
        inputs: ImageList,
        targets: Optional[List[Boxes2D]] = None,
    ) -> DetectionOutput:
        """Forward function."""
        if targets is not None:
            targets = target_to_instance(targets, inputs.tensor.shape[2:])

        # backbone
        feat = self.d2_detector.backbone(inputs.tensor)

        # rpn stage
        proposals, rpn_losses = self.d2_detector.proposal_generator(
            inputs, feat, targets
        )

        # detection head(s)
        detections, detect_losses = self.d2_detector.roi_heads(
            inputs,
            feat,
            proposals,
            targets,
        )

        proposals = proposal_to_box2d(proposals)
        if not self.d2_detector.training:
            detections = detections_to_box2d(detections)
        else:
            detections = proposal_to_box2d(detections)
        return feat, proposals, detections, {**rpn_losses, **detect_losses}
