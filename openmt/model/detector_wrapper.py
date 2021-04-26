"""Faster R-CNN for quasi-dense instance similarity learning."""
from typing import Dict, List, Tuple

import torch

from openmt.model.detect import BaseDetectorConfig, build_detector
from openmt.model.detect.d2_utils import target_to_box2d
from openmt.struct import Boxes2D

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
        self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        images = self.detector.preprocess_image(batch_inputs)  # type: ignore
        targets = [
            target_to_box2d(
                x["instances"].to(self.detector.device)  # type: ignore # pylint: disable=line-too-long
            )
            for x in batch_inputs
        ]

        # from openmt.vis.image import imshow_bboxes
        # for img, target in zip(images.tensor, targets):
        #     imshow_bboxes(img, target)

        _, _, _, det_losses = self.detector(images, targets)
        return det_losses  # type: ignore

    def forward_test(
        self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> List[Boxes2D]:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        images = self.detector.preprocess_image(batch_inputs)  # type: ignore
        _, _, detections, _ = self.detector(images)

        for inp, im, det in zip(batch_inputs, images, detections):
            self.postprocess(
                (inp["width"], inp["height"]),  # type: ignore
                (im.shape[-1], im.shape[-2]),
                det,
            )
        return detections  # type: ignore
