"""Panoptic FPN model."""
from typing import Dict, List, Union

from scalabel.label.typing import Label
from torch import nn

from vis4d.struct import ArgsType, InputSample, Losses, ModelOutput

from ..detect import BaseTwoStageDetector
from ..heads.dense_head import BaseSegmentationHead
from ..heads.panoptic_head import BasePanopticHead
from ..utils import postprocess_predictions


class PanopticFPN(nn.Module):
    """Panoptic FPN model."""

    def __init__(
        self,
        category_mapping: Dict[str, int],
        detection: BaseTwoStageDetector,
        seg_head: BaseSegmentationHead,
        pan_head: BasePanopticHead,
        *args: ArgsType,
        **kwargs: ArgsType
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.category_mapping = category_mapping
        self.detector = detection
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.detector.category_mapping = self.category_mapping
        self.seg_head = seg_head
        self.pan_head = pan_head
        self.det_mapping = {v: k for k, v in self.category_mapping.items()}

    @staticmethod
    def combine_segm_outs(
        ins_outs: List[List[Label]], sem_outs: List[List[Label]]
    ) -> List[List[Label]]:
        """Combine instance and semantic segmentation outputs."""
        return [
            ins_out + sem_out for ins_out, sem_out in zip(ins_outs, sem_outs)
        ]

    def forward_train(self, batch_inputs: List[InputSample]) -> Losses:
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
        outputs = dict(  # type: ignore
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
        model_outs["pan_seg"] = self.combine_segm_outs(
            model_outs["ins_seg"], model_outs["sem_seg"]
        )
        if getattr(self.pan_head, "ignore_class", -1) != -1:
            # set semantic segmentation prediction to panoptic segmentation
            model_outs["sem_seg"] = model_outs["pan_seg"]

        return model_outs

    def forward(
        self, batch_inputs: List[InputSample]
    ) -> Union[Losses, ModelOutput]:
        """Forward."""
        if self.training:
            return self.forward_train(batch_inputs)
        return self.forward_test(batch_inputs)
