"""Base classes for data structures in Vis4D."""
import abc
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from scalabel.label.typing import ImageSize, Label

TDataInstance = TypeVar("TDataInstance", bound="DataInstance")
TInputInstance = TypeVar("TInputInstance", bound="InputInstance")
TLabelInstance = TypeVar("TLabelInstance", bound="LabelInstance")

TTrainReturn = TypeVar("TTrainReturn")
TTestReturn = TypeVar("TTestReturn")


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
NamedTensors = Dict[str, torch.Tensor]


class Proposals(NamedTuple):
    """Output structure for object proposals."""

    boxes: List[torch.Tensor]
    scores: List[torch.Tensor]


class Detections(NamedTuple):
    boxes: torch.Tensor  # N, 4
    scores: torch.Tensor
    class_ids: torch.Tensor


class Tracks(NamedTuple):
    boxes: torch.Tensor  # N, 4
    scores: torch.Tensor
    class_ids: torch.Tensor
    track_ids: torch.Tensor


class Masks(NamedTuple):
    masks: torch.Tensor  # N, H, W
    scores: torch.Tensor
    class_ids: torch.Tensor
