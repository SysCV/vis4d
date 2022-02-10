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
from scalabel.label.typing import ImageSize, Label

TDataInstance = TypeVar("TDataInstance", bound="DataInstance")
TInputInstance = TypeVar("TInputInstance", bound="InputInstance")
TLabelInstance = TypeVar("TLabelInstance", bound="LabelInstance")


class DataInstance(metaclass=abc.ABCMeta):
    """Meta class for input data."""

    @abc.abstractmethod
    def to(  # pylint: disable=invalid-name
        self: "TDataInstance", device: torch.device
    ) -> "TDataInstance":
        """Move to device (CPU / GPU / ...)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return length of DataInstance."""
        raise NotImplementedError

    def __getitem__(self: "TDataInstance", item: int) -> "TDataInstance":
        """Return item of DataInstance."""
        raise NotImplementedError

    def __iter__(self: "TDataInstance") -> Iterator["TDataInstance"]:
        """Iterator definition of Images."""
        for i in range(len(self)):
            yield self[i]


class InputInstance(DataInstance, metaclass=abc.ABCMeta):
    """Interface for images, intrinsics, etc."""

    @classmethod
    @abc.abstractmethod
    def cat(
        cls: Type[TInputInstance],
        instances: List[TInputInstance],
        device: Optional[torch.device] = None,
    ) -> TInputInstance:
        """Concatenate multiple instances into a single one (batching)."""
        raise NotImplementedError


class LabelInstance(DataInstance, metaclass=abc.ABCMeta):
    """Interface for bounding boxes, masks etc."""

    @classmethod
    @abc.abstractmethod
    def empty(cls, device: Optional[torch.device] = None) -> "LabelInstance":
        """Return empty labels on device."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> "LabelInstance":
        """Convert from scalabel format to ours."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> List[Label]:
        """Convert from ours to scalabel format."""
        raise NotImplementedError

    def postprocess(
        self,
        original_wh: Tuple[int, int],
        output_wh: Tuple[int, int],
        clip: bool = True,
        resolve_overlap: bool = True,
    ) -> None:
        """Postprocess label according to original image resolution.

        Default behavior: Do nothing.
        """


NDArrayF64 = npt.NDArray[np.float64]
NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]
NDArrayUI8 = npt.NDArray[np.uint8]
TorchCheckpoint = Dict[str, Union[int, str, Dict[str, NDArrayF64]]]
LossesType = Dict[str, torch.Tensor]
ModelOutput = Dict[str, List[List[Label]]]
DictStrAny = Dict[str, Any]  # type: ignore
CategoryMap = Union[Dict[str, int], Dict[str, Dict[str, int]]]
MetricLogs = Dict[str, Union[float, int]]
FeatureMaps = Dict[str, torch.Tensor]
ModuleCfg = DictStrAny
ArgsType = Any  # type: ignore
