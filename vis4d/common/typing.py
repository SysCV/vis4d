"""Type definitions in Vis4D."""
from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
from torch import Tensor

NDArrayF64 = npt.NDArray[np.float64]
NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]
NDArrayUI8 = npt.NDArray[np.uint8]
MetricLogs = Dict[str, Union[float, int]]
DictStrAny = Dict[str, Any]  # type: ignore
ArgsType = Any  # type: ignore
ModelOutput = DictStrAny
TorchCheckpoint = Dict[str, Union[int, str, Dict[str, NDArrayF64]]]
LossesType = Dict[str, Tensor]


@dataclass
class MODEL_OUT_KEYS:
    """Container for common keys used in model outputs.

    Connects model outputs to evaluators, writers, etc.
    """

    boxes2d = "boxes2d"
    boxes2d_scores = "boxes2d_scores"
    boxes2d_classes = "boxes2d_classes"


_DictStrArray = Dict[str, Tensor]
_DictStrArrayNested = Dict[str, Union[Tensor, _DictStrArray]]
DictData = Dict[str, Union[Tensor, _DictStrArrayNested]]


@dataclass
class COMMON_KEYS:
    """Common supported keys for DictData.

    This container can hold arbitrary keys of data, where data of the keys defined
    here should be in the following format:

    original_hw: Tuple[int, int]
    input_hw: Tuple[int, int]
    transform_params: DictStrAny
    batch_transform_params: DictStrAny
    images: Tensor of shape [1, C, H, W]
    boxes2d: Tensor of shape [N, 4]
    boxes2d_classes: Tensor of shape [N,]
    masks: Tensor of shape [N, H, W]
    """

    original_hw = "original_hw"
    input_hw = "input_hw"
    transform_params = "transform_params"
    batch_transform_params = "batch_transform_params"
    images = "images"
    boxes2d = "boxes2d"
    boxes2d_classes = "boxes2d_classes"
    intrinsics = "intrinsics"
    extrinsics = "extrinsiscs"
    timestamp = "timestamp"
    masks = "masks"
    segmentation_mask = "segmentation_mask"
    points3d = "points3d"
    colors3d = "colors3d"
    semantics3d = "semantics3d"
    instances3d = "instances3d"
    boxes3d = "boxes3d"
    boxes3d_classes = "boxes3d_classes"
