"""Vis4D struct module."""
from .sample import InputData
from .structures import (
    ALLOWED_INPUTS,
    ALLOWED_TARGETS,
    ArgsType,
    CategoryMap,
    DictStrAny,
    FeatureMaps,
    Losses,
    MetricLogs,
    ModelOutput,
    NDArrayF32,
    NDArrayF64,
    NDArrayI64,
    NDArrayUI8,
    TorchCheckpoint,
)

__all__ = [
    "TorchCheckpoint",
    "NDArrayF64",
    "NDArrayF32",
    "NDArrayI64",
    "NDArrayUI8",
    "Losses",
    "ModelOutput",
    "DictStrAny",
    "MetricLogs",
    "FeatureMaps",
    "ArgsType",
    "CategoryMap",
    "ALLOWED_INPUTS",
    "ALLOWED_TARGETS",
]
