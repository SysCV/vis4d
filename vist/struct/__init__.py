"""OpenMT struct module."""
from .data import Images, InputSample
from .labels import Boxes2D, Boxes3D
from .structures import (
    DataInstance,
    DictStrAny,
    EvalResult,
    EvalResults,
    LabelInstance,
    LossesType,
    ModelOutput,
    NDArrayF64,
    NDArrayUI8,
    TorchCheckpoint,
)

__all__ = [
    "Boxes2D",
    "Boxes3D",
    "DataInstance",
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
