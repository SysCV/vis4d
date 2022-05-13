"""Base classes for data structures in Vis4D."""
import abc
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
import torch

ALLOWED_INPUTS = (
    "images",
    "intrinsics",
    "extrinsics",
    "pointcloud",
    "other",
)

ALLOWED_TARGETS = (
    "boxes2d",
    "boxes3d",
    "instance_masks",
    "semantic_masks",
    "other",
)

CategoryMap = Union[Dict[str, int], Dict[str, Dict[str, int]]]


NDArrayF64 = npt.NDArray[np.float64]
NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]
NDArrayUI8 = npt.NDArray[np.uint8]
TorchCheckpoint = Dict[str, Union[int, str, Dict[str, NDArrayF64]]]
Losses = Dict[str, torch.Tensor]
DictStrAny = Dict[str, Any]  # type: ignore
ModelOutput = DictStrAny  # TODO can this be constrained more? 'some nested dict with tensors / list tensors only'
MetricLogs = Dict[str, Union[float, int]]
FeatureMaps = Dict[str, torch.Tensor]
ArgsType = Any  # type: ignore
