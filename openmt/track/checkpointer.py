"""Checkpointing for tracking methods."""
from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.checkpoint import _IncompatibleKeys

from openmt.struct import TorchCheckpoint


class TrackingCheckpointer(DetectionCheckpointer):  # type: ignore
    """Tracking checkpointer.

    Loads detectron2 models into a tracking model.
    """

    def _load_model(self, checkpoint: TorchCheckpoint) -> _IncompatibleKeys:
        """Modify d2 checkpoint, load model weights."""
        # checkpoint modification to fit to d2_detector
        assert "__author__" in checkpoint.keys() and isinstance(
            checkpoint["__author__"], str
        )
        assert "model" in checkpoint.keys() and isinstance(
            checkpoint["model"], dict
        )
        if checkpoint["__author__"].startswith("Detectron2"):
            checkpoint["model"] = {
                "d2_detector." + k: v for k, v in checkpoint["model"].items()
            }
        incompatible = super()._load_model(checkpoint)
        return incompatible
