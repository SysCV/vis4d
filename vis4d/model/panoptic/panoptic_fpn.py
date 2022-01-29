"""Panoptic FPN.

Panoptic Feature Pyramid Networks
https://arxiv.org/abs/1901.02446
"""
from typing import Dict, List, Union

from vis4d.common.module import build_module
from vis4d.struct import (
    ArgsType,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    ModuleCfg,
    TLabelInstance,
)

from ..base import BaseModel, build_model
from ..detect import BaseTwoStageDetector
from ..heads.dense_head import BaseDenseHead, SegDenseHead
from ..heads.panoptic_head import BasePanopticHead
from ..utils import (
    combine_pan_outs,
    postprocess_predictions,
    predictions_to_scalabel,
)


class PanopticFPN(BaseModel):
    """Panoptic FPN model."""

    def __init__(
        self,
        detection: Union[BaseTwoStageDetector, ModuleCfg],
        seg_head: Union[SegDenseHead, ModuleCfg],
        pan_head: Union[BasePanopticHead, ModuleCfg],
        *args: ArgsType,
        **kwargs: ArgsType
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None
        if isinstance(detection, dict):
            detection["category_mapping"] = self.category_mapping
            self.detector: BaseTwoStageDetector = build_model(detection)
        else:  # pragma: no cover
            self.detector = detection
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.detector.category_mapping = self.category_mapping
        if isinstance(seg_head, dict):
            self.seg_head: SegDenseHead = build_module(
                seg_head, bound=BaseDenseHead
            )
        else:  # pragma: no cover
            self.seg_head = seg_head
        if isinstance(pan_head, dict):
            self.pan_head: BasePanopticHead = build_module(
                pan_head, bound=BasePanopticHead
            )
        else:  # pragma: no cover
            self.pan_head = pan_head
        self.det_mapping = {v: k for k, v in self.category_mapping.items()}

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in PanopticFPN training!"
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
        ), "No reference views allowed in PanopticFPN testing!"
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
            inputs,
            outputs,
            clip_to_image=self.detector.clip_bboxes_to_image,
            resolve_overlap=False,
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

        # combine ins_seg and sem_seg to get pan_seg predictions
        model_outs["pan_seg"] = combine_pan_outs(
            model_outs["ins_seg"], model_outs["sem_seg"]
        )
        if getattr(self.pan_head, "ignore_class", -1) != -1:
            # set semantic segmentation prediction to panoptic segmentation
            model_outs["sem_seg"] = model_outs["pan_seg"]

        return model_outs
