"""Common type definitions.

Here we define commonly used types like specific numpy array and tensor types.
"""
from typing import Any, Dict, Iterable, Union

import numpy as np
import numpy.typing as npt
from torch import (  # pylint: disable=no-name-in-module
    BoolTensor,
    FloatTensor,
    IntTensor,
    Tensor,
)

NumpyFloat = Union[np.float32, np.float64]
NumpyInt = Union[np.int32, np.int64]
NumpyBool = np.bool_

NDArrayF64 = npt.NDArray[np.float64]
NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]
NDArrayI32 = npt.NDArray[np.int32]
NDArrayUI8 = npt.NDArray[np.uint8]
NDArrayBool = npt.NDArray[np.bool_]
NDArrayFloat = Union[NDArrayF32, NDArrayF64]

NDArrayInt = Union[NDArrayI64, NDArrayI32]

NDArrayNumber = Union[NDArrayFloat, NDArrayInt, NDArrayBool]
MetricLogs = Dict[str, Union[float, int]]
DictStrAny = Dict[str, Any]  # type: ignore
ArgsType = Any  # type: ignore
ModelOutput = DictStrAny
TorchCheckpoint = Dict[str, Union[int, str, Dict[str, NDArrayF64]]]
LossesType = Dict[str, Tensor]

ArrayIterableInt = Iterable[Union[int, "ArrayIterableInt"]]
ArrayIterableFloat = Iterable[Union[float, "ArrayIterableFloat"]]
ArrayIterableBool = Iterable[Union[bool, "ArrayIterableBool"]]

ArrayLikeFloat = Union[ArrayIterableFloat, NDArrayF32, NDArrayF64, FloatTensor]
ArrayLikeBool = Union[ArrayIterableBool, NDArrayBool, BoolTensor]
ArrayLikeInt = Union[ArrayIterableInt, NDArrayInt, IntTensor]
ArrayLike = Union[ArrayLikeBool, ArrayLikeFloat, ArrayLikeInt]
