"""Panoptic segmentation model."""
from typing import List

from vis4d.struct import InputSample, LabelInstances, LossesType, ModelOutput

from .base import BaseModel, BaseModelConfig, build_model
from .detect import BaseDetectorConfig, BaseTwoStageDetector
from .heads.dense_head import (
    BaseDenseHeadConfig,
    MMSegDecodeHead,
    build_dense_head,
)
from .heads.panoptic_head import (
    BasePanopticHead,
    BasePanopticHeadConfig,
    build_panoptic_head,
)


class PanopticSegmentorConfig(BaseModelConfig):
    """Config for panoptic segmentation model."""

    detection: BaseDetectorConfig
    seg_head: BaseDenseHeadConfig
    pan_head: BasePanopticHeadConfig


class PanopticSegmentor(BaseModel):
    """Panoptic segmentation model."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__(cfg)
        self.cfg: PanopticSegmentorConfig = PanopticSegmentorConfig(
            **cfg.dict()
        )
        self.detector: BaseTwoStageDetector = build_model(self.cfg.detection)
        self.seg_head: MMSegDecodeHead = build_dense_head(self.cfg.seg_head)
        self.pan_head: BasePanopticHead = build_panoptic_head(
            self.cfg.pan_head
        )

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in PanopticSegmentor training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.backbone(inputs)
        rpn_losses, proposals = self.detector.generate_proposals(
            inputs, features, targets
        )
        roi_losses, _ = self.detector.generate_detections(
            inputs, features, proposals, targets
        )
        seg_losses, _ = self.seg_head(inputs, features, targets)
        return {**rpn_losses, **roi_losses, **seg_losses}

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in PanopticSegmentor testing!"
        inputs = batch_inputs[0]
        feat = self.detector.extract_features(inputs)
        proposals = self.detector.generate_proposals(inputs, feat)
        detections, instance_segms = self.detector.generate_detections(
            inputs, feat, proposals
        )
        semantic_segms = self.seg_head(inputs, feat)

        for inp, det in zip(inputs, detections):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            det.postprocess(
                input_size,
                inp.images.image_sizes[0],
                self.cfg.detection.clip_bboxes_to_image,
            )

        predictions = LabelInstances(
            detections,
            instance_masks=instance_segms,
            semantic_masks=semantic_segms,
        )
        panoptic_segms = self.pan_head(inputs, predictions)

        return outputs
