"""Base classes for data structures in VisT."""
import abc
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from scalabel.eval.mot import EvalResults as MOTEvalResults
from scalabel.label.typing import Label


class DataInstance(metaclass=abc.ABCMeta):
    """Meta class for input data."""

    @abc.abstractmethod
    def to(  # pylint: disable=invalid-name
        self, device: torch.device
    ) -> "DataInstance":
        """Move to device (CPU / GPU / ...)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        raise NotImplementedError


class LabelInstance(DataInstance, metaclass=abc.ABCMeta):
    """Interface for bounding boxes, masks etc."""

    @classmethod
    @abc.abstractmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
    ) -> "LabelInstance":
        """Convert from scalabel format to ours."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_scalabel(self, idx_to_class: Dict[int, str]) -> List[Label]:
        """Convert from ours to scalabel format."""
        raise NotImplementedError


NDArrayF64 = npt.NDArray[np.float64]
NDArrayUI8 = npt.NDArray[np.uint8]
TorchCheckpoint = Dict[str, Union[int, str, Dict[str, NDArrayF64]]]
LossesType = Dict[str, torch.Tensor]
EvalResult = Union[Dict[str, float], MOTEvalResults]
EvalResults = Dict[str, Union[Dict[str, float], MOTEvalResults]]
ModelOutput = Dict[str, List[LabelInstance]]
DictStrAny = Dict[str, Any]  # type: ignore
