"""OpenMT struct module."""
from .data import (
    DictStrAny,
    EvalResult,
    EvalResults,
    Images,
    InputSample,
    LossesType,
    NDArrayF64,
    NDArrayUI8,
    TorchCheckpoint,
)
from .labels import Boxes2D, LabelInstance, ModelOutput

__all__ = [
    "Boxes2D",
    "LabelInstance",
    "TorchCheckpoint",
    "NDArrayF64",
    "NDArrayUI8",
    "LossesType",
    "EvalResult",
    "EvalResults",
    "InputSample",
    "Images",
    "ModelOutput",
    "DictStrAny",
]
