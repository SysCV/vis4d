"""Checkpointing for tracking methods."""
from typing import Dict

from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.checkpoint import _IncompatibleKeys


class TrackingCheckpointer(DetectionCheckpointer):
    """Tracking checkpointer.
    Loads detectron2 models into a tracking model.
    """

    def _load_model(self, checkpoint: Dict) -> _IncompatibleKeys:
        """Modify d2 checkpoint, load model weights."""
        # checkpoint modification to fit to d2_detector
        if checkpoint["__author__"].startswith("Detectron2"):
            checkpoint["model"] = {
                "d2_detector." + k: v for k, v in checkpoint["model"].items()
            }
        incompatible = super()._load_model(checkpoint)
        return incompatible
