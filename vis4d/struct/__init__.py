"""Vis4D struct module."""
from .data import Extrinsics, Images, Intrinsics
from .labels import (
    Boxes2D,
    Boxes3D,
    InstanceMasks,
    Masks,
    SemanticMasks,
    TMasks,
)
from .sample import InputSample, LabelInstances
from .structures import (
    DataInstance,
    DictStrAny,
    FeatureMaps,
    LabelInstance,
    LossesType,
    MetricLogs,
    ModelOutput,
    NDArrayF64,
    NDArrayI64,
    NDArrayUI8,
    TLabelInstance,
    TorchCheckpoint,
)

__all__ = [
    "Boxes2D",
    "Boxes3D",
    "Masks",
    "TMasks",
    "InstanceMasks",
    "SemanticMasks",
    "DataInstance",
    "LabelInstance",
    "TorchCheckpoint",
    "NDArrayF64",
    "NDArrayI64",
    "NDArrayUI8",
    "LossesType",
    "Images",
    "Intrinsics",
    "Extrinsics",
    "ModelOutput",
    "DictStrAny",
    "InputSample",
    "LabelInstances",
    "TLabelInstance",
    "MetricLogs",
    "FeatureMaps",
]
