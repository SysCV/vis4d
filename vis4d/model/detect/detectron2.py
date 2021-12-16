"""Detectron2 detector wrapper."""
from typing import Dict, List, Optional, Tuple

import torch

try:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import GeneralizedRCNN
    from detectron2.structures import Instances
    from detectron2.utils.events import EventStorage

    from .d2_utils import (
        D2TwoStageDetectorConfig,
        box2d_to_proposal,
        detections_to_box2d,
        images_to_imagelist,
        model_to_detectron2,
        proposal_to_box2d,
        segmentations_to_bitmask,
        target_to_instance,
    )

    D2_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    D2_INSTALLED = False

from torch.nn.modules.batchnorm import _BatchNorm

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    ModelOutput,
    TLabelInstance,
)

from ..utils import predictions_to_scalabel
from .base import BaseTwoStageDetector


class D2TwoStageDetector(BaseTwoStageDetector):
    """Detectron2 two-stage detector wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        assert (
            D2_INSTALLED
        ), "D2TwoStageDetector requires detectron2 to be installed!"
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

    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Batch, pad (standard stride=32) and normalize the input images."""
        inputs.images.tensor = (
            inputs.images.tensor - self.d2_detector.pixel_mean
        ) / self.d2_detector.pixel_std
        return inputs

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """D2 model forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in D2TwoStageDetector training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        inputs = self.preprocess_inputs(inputs)
        features = self.extract_features(inputs)
        rpn_losses, proposals = self.generate_proposals(
            inputs, features, targets
        )
        detect_losses, _ = self.generate_detections(
            inputs, features, proposals, targets
        )
        return {**rpn_losses, **detect_losses}

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in D2TwoStageDetector testing!"
        inputs = self.preprocess_inputs(batch_inputs[0])
        features = self.extract_features(inputs)
        proposals = self.generate_proposals(inputs, features)
        detections, segmentations = self.generate_detections(
            inputs, features, proposals
        )

        outputs: Dict[str, List[TLabelInstance]] = dict(detect=detections)  # type: ignore # pylint: disable=line-too-long
        if self.with_mask:
            assert segmentations is not None
            outputs["ins_seg"] = segmentations

        return predictions_to_scalabel(
            inputs, outputs, self.cat_mapping, self.cfg.clip_bboxes_to_image
        )

    def extract_features(self, inputs: InputSample) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return backbone output features.
        """
        return self.d2_detector.backbone(inputs.images.tensor)  # type: ignore

    def _proposals_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, List[Boxes2D]]:
        """Train stage proposal generation."""
        images_d2 = images_to_imagelist(inputs.images)
        targets_d2: Optional[List[Instances]] = target_to_instance(
            inputs.targets.boxes2d, inputs.images.image_sizes
        )
        self.d2_detector.proposal_generator.training = True

        with self.d2_event_storage:
            proposals, rpn_losses = self.d2_detector.proposal_generator(
                images_d2, features, targets_d2
            )
        return rpn_losses, proposal_to_box2d(proposals)

    def _proposals_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> List[Boxes2D]:
        """Test stage proposal generation."""
        images_d2 = images_to_imagelist(inputs.images)
        targets_d2 = None
        self.d2_detector.proposal_generator.training = False

        with self.d2_event_storage:
            proposals, _ = self.d2_detector.proposal_generator(
                images_d2, features, targets_d2
            )
        return proposal_to_box2d(proposals)

    def _detections_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Train stage detections generation."""
        images_d2 = images_to_imagelist(inputs.images)
        proposals = box2d_to_proposal(proposals, inputs.images.image_sizes)
        targets_d2: Optional[List[Instances]] = target_to_instance(
            inputs.targets.boxes2d,
            inputs.images.image_sizes,
            inputs.targets.instance_masks,
        )
        self.d2_detector.roi_heads.training = True

        with self.d2_event_storage:
            _, detect_losses = self.d2_detector.roi_heads(
                images_d2,
                features,
                proposals,
                targets_d2,
            )
        return detect_losses, None

    def _detections_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:
        """Test stage detections generation."""
        images_d2 = images_to_imagelist(inputs.images)
        proposals = box2d_to_proposal(proposals, inputs.images.image_sizes)
        targets_d2 = None
        self.d2_detector.roi_heads.training = False

        with self.d2_event_storage:
            detections, _ = self.d2_detector.roi_heads(
                images_d2,
                features,
                proposals,
                targets_d2,
            )

        segmentations: Optional[List[InstanceMasks]] = None
        dets_boxes2d = detections_to_box2d(detections)
        if self.with_mask:
            segmentations = segmentations_to_bitmask(detections, dets_boxes2d)
        return dets_boxes2d, segmentations
