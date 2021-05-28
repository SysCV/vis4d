"""Wrapper for training / testing detectors in openMT."""
from typing import List

from openmt.model.detect import BaseDetectorConfig, build_detector
from openmt.struct import InputSample, LossesType, ModelOutput

from .base import BaseModel, BaseModelConfig


class DetectorWrapperConfig(BaseModelConfig):
    """Config for detection wrapper."""

    detection: BaseDetectorConfig


class DetectorWrapper(BaseModel):
    """Wrapper model for a detector."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = DetectorWrapperConfig(**cfg.dict())
        self.detector = build_detector(self.cfg.detection)

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        inputs = [inp[0] for inp in batch_inputs]  # no ref views

        # from openmt.vis.image import imshow_bboxes
        # for inp in inputs:
        #     imshow_bboxes(inp.image.tensor[0], inp.instances)

        targets = [x.instances.to(self.detector.device) for x in inputs]
        _, _, _, _, det_losses = self.detector(inputs, targets)
        return det_losses  # type: ignore

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        _, _, _, detections, _ = self.detector(batch_inputs)
        for inp, det in zip(batch_inputs, detections):
            ori_wh = (
                batch_inputs[0].metadata.size.width,  # type: ignore
                batch_inputs[0].metadata.size.height,  # type: ignore
            )
            self.postprocess(ori_wh, inp.image.image_sizes[0], det)
        return dict(detect=detections)
