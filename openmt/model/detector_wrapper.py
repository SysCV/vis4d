"""Faster R-CNN for quasi-dense instance similarity learning."""
from typing import Dict, List, Tuple

import torch

from openmt.data.utils import target_to_box2d
from openmt.model.detect import BaseDetectorConfig, build_detector
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
            x["instances"].to(self.detector.device)  # type: ignore
            for x in batch_inputs
        ]
        targets = [
            target_to_box2d(target, score_as_logit=False) for target in targets
        ]
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
        return detections  # type: ignore
