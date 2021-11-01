"""Detectron2 detector wrapper."""
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import GeneralizedRCNN
from detectron2.structures import Instances
from detectron2.utils.events import EventStorage
from torch.nn.modules.batchnorm import _BatchNorm

from vist.model.detect.d2_utils import (
    D2TwoStageDetectorConfig,
    box2d_to_proposal,
    detections_to_box2d,
    images_to_imagelist,
    model_to_detectron2,
    proposal_to_box2d,
    segmentations_to_bitmask,
    target_to_instance,
)
from vist.struct import (
    Boxes2D,
    InputSample,
    InstanceMasks,
    LossesType,
    ModelOutput,
)

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
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.d2_cfg = model_to_detectron2(self.cfg)
        # pylint: disable=too-many-function-args,missing-kwoa
        self.d2_detector = GeneralizedRCNN(self.d2_cfg)
        # detectron2 requires an EventStorage for logging
        self.d2_event_storage = EventStorage()
        self.checkpointer = DetectionCheckpointer(self.d2_detector)
        self.with_mask = self.d2_detector.roi_heads.mask_on
        if self.d2_cfg.MODEL.WEIGHTS != "":
            self.checkpointer.load(self.d2_cfg.MODEL.WEIGHTS)
        if self.cfg.set_batchnorm_eval:
            self.set_batchnorm_eval()

    def set_batchnorm_eval(self) -> None:
        """Set all batchnorm layers in backbone to eval mode."""
        for m in self.d2_detector.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def preprocess_inputs(self, inputs: List[InputSample]) -> InputSample:
        """Batch, pad (standard stride=32) and normalize the input images."""
        batched_inputs = InputSample.cat(inputs, self.device)
        batched_inputs.images.tensor = (
            batched_inputs.images.tensor - self.d2_detector.pixel_mean
        ) / self.d2_detector.pixel_std
        return batched_inputs

    def forward_train(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """D2 model forward pass during training stage."""
        assert all(
            len(inp) == 1 for inp in batch_inputs
        ), "No reference views allowed in D2TwoStageDetector training!"
        raw_inputs = [inp[0] for inp in batch_inputs]

        inputs = self.preprocess_inputs(raw_inputs)
        features = self.extract_features(inputs)
        proposals, rpn_losses = self.generate_proposals(inputs, features)
        _, detect_losses, _ = self.generate_detections(
            inputs, features, proposals, compute_detections=False
        )
        return {**rpn_losses, **detect_losses}

    def forward_test(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> ModelOutput:
        """Forward pass during testing stage."""
        raw_inputs = [inp[0] for inp in batch_inputs]
        inputs = self.preprocess_inputs(raw_inputs)
        features = self.extract_features(inputs)
        proposals, _ = self.generate_proposals(inputs, features)
        detections, _, segmentations = self.generate_detections(
            inputs, features, proposals
        )
        assert detections is not None
        if segmentations is None:
            segmentations = [None] * len(detections)  # type: ignore

        for inp, det, segm in zip(inputs, detections, segmentations):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            det.postprocess(input_size, inp.images.image_sizes[0])
            if segm is not None:
                segm.postprocess(input_size, inp.images.image_sizes[0], det)

        outputs = dict(
            detect=[d.to_scalabel(self.cat_mapping) for d in detections]
        )
        if self.with_mask:
            outputs.update(
                ins_seg=[
                    s.to_scalabel(self.cat_mapping) for s in segmentations
                ]
            )
        return outputs

    def extract_features(self, inputs: InputSample) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return backbone output features.
        """
        return self.d2_detector.backbone(inputs.images.tensor)  # type: ignore

    def generate_proposals(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        images_d2 = images_to_imagelist(inputs.images)
        is_training = self.d2_detector.proposal_generator.training
        if self.training:
            targets: Optional[List[Instances]] = target_to_instance(
                inputs.boxes2d, inputs.images.image_sizes
            )
        else:
            targets = None
            self.d2_detector.proposal_generator.training = False

        with self.d2_event_storage:
            proposals, rpn_losses = self.d2_detector.proposal_generator(
                images_d2, features, targets
            )
        self.d2_detector.proposal_generator.training = is_training
        return proposal_to_box2d(proposals), rpn_losses

    def generate_detections(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        proposals: Optional[List[Boxes2D]] = None,
        compute_detections: bool = True,
        compute_segmentations: bool = False,
    ) -> Tuple[
        Optional[List[Boxes2D]], LossesType, Optional[List[InstanceMasks]]
    ]:
        """Detector second stage (RoI Head).

        Return losses (empty if no targets) and optionally detections.
        """
        assert (
            proposals is not None
        ), "Generating detections with D2TwoStageDetector requires proposals."
        images_d2 = images_to_imagelist(inputs.images)
        proposals = box2d_to_proposal(proposals, inputs.images.image_sizes)
        is_training = self.d2_detector.roi_heads.training
        if self.training:
            targets: Optional[List[Instances]] = target_to_instance(
                inputs.boxes2d,
                inputs.images.image_sizes,
                inputs.instance_masks,
            )
        else:
            targets = None
            self.d2_detector.roi_heads.training = False

        with self.d2_event_storage:
            detections, detect_losses = self.d2_detector.roi_heads(
                images_d2,
                features,
                proposals,
                targets,
            )
        self.d2_detector.roi_heads.training = is_training
        segmentations = None
        if not self.d2_detector.training:
            if self.with_mask:
                segmentations = segmentations_to_bitmask(detections)
            detections = detections_to_box2d(detections)
        elif compute_detections:  # pragma: no cover
            if compute_segmentations:
                segmentations = segmentations_to_bitmask(detections)
            detections = proposal_to_box2d(detections)
        else:
            detections = None
        return detections, detect_losses, segmentations
