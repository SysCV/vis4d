"""Type definitions in Vis4D."""
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
