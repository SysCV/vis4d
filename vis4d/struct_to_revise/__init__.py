"""Vis4D struct module."""
from .data import Extrinsics, Images, Intrinsics, PointCloud
from .labels import Boxes2D, Boxes3D, InstanceMasks, SemanticMasks, TMasks
from .sample import InputSample, LabelInstances
from .structures import (
    ALLOWED_INPUTS,
    ALLOWED_TARGETS,
    CategoryMap,
    DataInstance,
    Detections,
    LabelInstance,
    Masks,
    NamedTensors,
    Proposals,
    TLabelInstance,
    Tracks,
    TTestReturn,
    TTrainReturn,
)

__all__ = [
    "Detections",
    "Tracks",
    "Proposals",
    "Boxes2D",
    "Boxes3D",
    "Masks",
    "TMasks",
    "InstanceMasks",
    "SemanticMasks",
    "DataInstance",
    "LabelInstance",
    "Images",
    "Intrinsics",
    "Extrinsics",
    "InputSample",
    "PointCloud",
    "LabelInstances",
    "TLabelInstance",
    "NamedTensors",
    "CategoryMap",
    "ALLOWED_INPUTS",
    "ALLOWED_TARGETS",
    "TTrainReturn",
    "TTestReturn",
]
