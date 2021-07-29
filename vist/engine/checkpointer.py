"""Checkpointing for openMT methods."""
from detectron2.checkpoint import DetectionCheckpointer as D2Checkpointer
from fvcore.common.checkpoint import _IncompatibleKeys

from openmt.struct import TorchCheckpoint


class Checkpointer(D2Checkpointer):  # type: ignore
    """OpenMT interface for D2 checkpoints.

    Loads both openmt and detectron2 models.
    """

    def _load_model(self, checkpoint: TorchCheckpoint) -> _IncompatibleKeys:
        """Modify d2 checkpoint / load weights."""
        assert "model" in checkpoint.keys() and isinstance(
            checkpoint["model"], dict
        )
        # checkpoint modification to fit to detector.d2_detector
        if "__author__" in checkpoint.keys():
            assert isinstance(checkpoint["__author__"], str)
            if checkpoint["__author__"].startswith("Detectron2"):
                checkpoint["model"] = {
                    "retinanet." + k: v for k, v in checkpoint["model"].items()
                }
        incompatible = super()._load_model(checkpoint)
        return incompatible
