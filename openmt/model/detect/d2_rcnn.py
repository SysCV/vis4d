"""Detectron2 detector wrapper."""
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.modeling import GeneralizedRCNN

from openmt.model.detect.d2_utils import (
    D2GeneralizedRCNNConfig,
    box2d_to_proposal,
    detections_to_box2d,
    images_to_imagelist,
    model_to_detectron2,
    proposal_to_box2d,
    target_to_instance,
)
from openmt.struct import (
    Boxes2D,
    DetectionOutput,
    Images,
    InputSample,
    LossesType,
)

from .base import BaseDetectorConfig, BaseTwoStageDetector


class D2GeneralizedRCNN(BaseTwoStageDetector):
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

    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Batch, pad (standard stride=32) and normalize the input images."""
        images = Images.cat([inp.image for inp in batched_inputs])
        images = images.to(self.device)
        images.tensor = (
            images.tensor - self.d2_detector.pixel_mean
        ) / self.d2_detector.pixel_std
        return images

    def forward(
        self,
        inputs: List[InputSample],
        targets: Optional[List[Boxes2D]] = None,
    ) -> DetectionOutput:
        """Forward function."""
        images = self.preprocess_image(inputs)
        features = self.extract_features(images)
        proposals, rpn_losses = self.generate_proposals(
            images, features, targets
        )
        detections, detect_losses = self.generate_detections(
            images, features, proposals, targets
        )
        return (
            images,
            features,
            proposals,
            detections,
            {**rpn_losses, **detect_losses},
        )

    def extract_features(self, images: Images) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return preprocessed images, backbone output features.
        """
        return self.d2_detector.backbone(images.tensor)  # type: ignore

    def generate_proposals(
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Boxes2D]] = None,
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        images_d2 = images_to_imagelist(images)
        is_training = self.d2_detector.proposal_generator.training
        if targets is not None:
            targets = target_to_instance(targets, images.image_sizes)
        else:
            self.d2_detector.proposal_generator.training = False

        proposals, rpn_losses = self.d2_detector.proposal_generator(
            images_d2, features, targets
        )
        self.d2_detector.proposal_generator.training = is_training
        return proposal_to_box2d(proposals), rpn_losses

    def generate_detections(
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector second stage (RoI Head).

        Return detections per image and losses (empty if no targets).
        """
        images_d2 = images_to_imagelist(images)
        proposals = box2d_to_proposal(proposals, images.image_sizes)
        is_training = self.d2_detector.roi_heads.training
        if targets is not None:
            targets = target_to_instance(targets, images.image_sizes)
        else:
            self.d2_detector.roi_heads.training = False

        detections, detect_losses = self.d2_detector.roi_heads(
            images_d2,
            features,
            proposals,
            targets,
        )
        self.d2_detector.roi_heads.training = is_training
        if not self.d2_detector.training:
            detections = detections_to_box2d(detections)
        else:
            detections = proposal_to_box2d(detections)
        return detections, detect_losses
