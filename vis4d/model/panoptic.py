"""Panoptic segmentation model."""
from typing import Dict, List

from vis4d.struct import (
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    TLabelInstance,
)

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
from .utils import postprocess_predictions, predictions_to_scalabel


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
        assert self.cfg.category_mapping is not None
        self.cfg.detection.category_mapping = self.cfg.category_mapping
        self.detector: BaseTwoStageDetector = build_model(self.cfg.detection)
        self.seg_head: MMSegDecodeHead = build_dense_head(self.cfg.seg_head)
        self.pan_head: BasePanopticHead = build_panoptic_head(
            self.cfg.pan_head
        )
        self.det_mapping = {
            v: k for k, v in self.cfg.detection.category_mapping.items()
        }

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in PanopticSegmentor training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.detector.extract_features(inputs)
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

        assert instance_segms is not None
        outputs: Dict[str, List[TLabelInstance]] = dict(  # type: ignore
            detect=detections, ins_seg=instance_segms, sem_seg=semantic_segms
        )
        postprocess_predictions(
            inputs, outputs, self.cfg.detection.clip_bboxes_to_image
        )

        predictions = LabelInstances(
            detections,
            instance_masks=instance_segms,
            semantic_masks=semantic_segms,
        )
        instance_segms, semantic_segms = self.pan_head(inputs, predictions)

        outputs.pop("sem_seg")
        model_outs = predictions_to_scalabel(outputs, self.det_mapping)
        # sem_seg has different category_mapping
        model_outs.update(
            predictions_to_scalabel(
                {"sem_seg": semantic_segms}, self.seg_head.cat_mapping
            )
        )
        return model_outs
