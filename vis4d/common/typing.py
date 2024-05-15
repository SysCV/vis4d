"""Common type definitions.

Here we define commonly used types like specific numpy array and tensor types.
"""

from collections.abc import Callable
from typing import Any, Dict, Iterable, Union

import numpy as np
import numpy.typing as npt
from torch import (  # pylint: disable=no-name-in-module
    BoolTensor,
    ByteTensor,
    FloatTensor,
    IntTensor,
    Tensor,
)

NumpyBool = np.bool_
NumpyFloat = Union[np.float32, np.float64]
NumpyInt = Union[np.int32, np.int64]
NumpyUInt = Union[  # pylint: disable=invalid-name
    np.uint8, np.uint16, np.uint32
]

NDArrayBool = npt.NDArray[np.bool_]
NDArrayF32 = npt.NDArray[np.float32]
NDArrayF64 = npt.NDArray[np.float64]
NDArrayFloat = Union[NDArrayF32, NDArrayF64]
NDArrayI32 = npt.NDArray[np.int32]
NDArrayI64 = npt.NDArray[np.int64]
NDArrayInt = Union[NDArrayI32, NDArrayI64]
NDArrayUI8 = npt.NDArray[np.uint8]
NDArrayUI16 = npt.NDArray[np.uint16]
NDArrayUI32 = npt.NDArray[np.uint32]
NDArrayUInt = Union[  # pylint: disable=invalid-name
    NDArrayUI8, NDArrayUI16, NDArrayUI32
]
NDArrayNumber = Union[NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayUInt]

MetricLogs = Dict[str, Union[float, int, Tensor]]
DictStrAny = Dict[str, Any]  # type: ignore
DictStrArrNested = Dict[str, Union[Tensor, Dict[str, Tensor]]]
ArgsType = Any  # type: ignore
ModelOutput = DictStrAny
TorchCheckpoint = DictStrAny
LossesType = Dict[str, Tensor]
TorchLossFunc = Callable[..., Any]  # type: ignore
GenericFunc = Callable[..., Any]  # type: ignore
TrainingModule = Any  # type: ignore

ArrayIterableFloat = Iterable[Union[float, "ArrayIterableFloat"]]
ArrayIterableBool = Iterable[Union[bool, "ArrayIterableBool"]]
ArrayIterableInt = Iterable[Union[int, "ArrayIterableInt"]]
ArrayIterableUInt = Iterable[Union[int, "ArrayIterableUInt"]]

ArrayLikeFloat = Union[ArrayIterableFloat, NDArrayF32, NDArrayF64, FloatTensor]
ArrayLikeBool = Union[ArrayIterableBool, NDArrayBool, BoolTensor]
ArrayLikeInt = Union[ArrayIterableInt, NDArrayInt, IntTensor]
ArrayLikeUInt = Union[  # pylint: disable=invalid-name
    ArrayIterableUInt, NDArrayUInt, ByteTensor
]
ArrayLike = Union[ArrayLikeBool, ArrayLikeFloat, ArrayLikeInt, ArrayLikeUInt]

ListAny = list[Any]  # type: ignore
