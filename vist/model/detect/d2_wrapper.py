"""Detectron2 detector wrapper."""
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.events import EventStorage
from torch.nn.modules.batchnorm import _BatchNorm

from vist.model.detect.d2_utils import (
    D2TwoStageDetectorConfig,
    box2d_to_proposal,
    detections_to_box2d,
    images_to_imagelist,
    model_to_detectron2,
    proposal_to_box2d,
    target_to_instance,
)
from vist.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput

from ..base import BaseModelConfig
from .base import BaseTwoStageDetector


class D2TwoStageDetector(BaseTwoStageDetector):
    """Detectron2 two-stage detector wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__(cfg)
        self.cfg = D2TwoStageDetectorConfig(
            **cfg.dict()
        )  # type: D2TwoStageDetectorConfig
        self.d2_cfg = model_to_detectron2(self.cfg)
        # pylint: disable=too-many-function-args,missing-kwoa
        self.d2_detector = GeneralizedRCNN(self.d2_cfg)
        # detectron2 requires an EventStorage for logging
        self.d2_event_storage = EventStorage()
        self.checkpointer = DetectionCheckpointer(self.d2_detector)
        if self.d2_cfg.MODEL.WEIGHTS != "":
            self.checkpointer.load(self.d2_cfg.MODEL.WEIGHTS)
        if self.cfg.set_batchnorm_eval:
            self.set_batchnorm_eval()

    def set_batchnorm_eval(self) -> None:
        """Set all batchnorm layers in backbone to eval mode."""
        for m in self.d2_detector.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Batch, pad (standard stride=32) and normalize the input images."""
        images = Images.cat([inp.image for inp in batched_inputs], self.device)
        images.tensor = (
            images.tensor - self.d2_detector.pixel_mean
        ) / self.d2_detector.pixel_std
        return images

    def forward_train(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """D2 model forward pass during training stage."""
        assert all(
            len(inp) == 1 for inp in batch_inputs
        ), "No reference views allowed in D2TwoStageDetector training!"
        inputs = [inp[0] for inp in batch_inputs]

        targets = []
        for x in inputs:
            assert x.boxes2d is not None
            targets.append(x.boxes2d.to(self.device))

        images = self.preprocess_image(inputs)
        features = self.extract_features(images)
        proposals, rpn_losses = self.generate_proposals(
            images, features, targets
        )
        _, detect_losses = self.generate_detections(
            images, features, proposals, targets, compute_detections=False
        )
        return {**rpn_losses, **detect_losses}

    def forward_test(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> ModelOutput:
        """Forward pass during testing stage."""
        inputs = [inp[0] for inp in batch_inputs]
        images = self.preprocess_image(inputs)
        features = self.extract_features(images)
        proposals, _ = self.generate_proposals(images, features)
        detections, _ = self.generate_detections(images, features, proposals)
        assert detections is not None
        assert inputs[0].metadata.size is not None
        input_size = (
            inputs[0].metadata.size.width,
            inputs[0].metadata.size.height,
        )
        for inp, det in zip(inputs, detections):
            self.postprocess(input_size, inp.image.image_sizes[0], det)

        return dict(detect=detections)  # type: ignore

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

        with self.d2_event_storage:
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
        compute_detections: bool = True,
    ) -> Tuple[Optional[List[Boxes2D]], LossesType]:
        """Detector second stage (RoI Head).

        Return losses (empty if no targets) and optionally detections.
        """
        images_d2 = images_to_imagelist(images)
        proposals = box2d_to_proposal(proposals, images.image_sizes)
        is_training = self.d2_detector.roi_heads.training
        if targets is not None:
            targets = target_to_instance(targets, images.image_sizes)
        else:
            self.d2_detector.roi_heads.training = False

        with self.d2_event_storage:
            detections, detect_losses = self.d2_detector.roi_heads(
                images_d2,
                features,
                proposals,
                targets,
            )
        self.d2_detector.roi_heads.training = is_training
        if not self.d2_detector.training:
            detections = detections_to_box2d(detections)
        elif compute_detections:
            detections = proposal_to_box2d(detections)  # pragma: no cover
        else:
            detections = None
        return detections, detect_losses
