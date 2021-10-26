"""Input sample definition in Vist."""
from typing import Dict, List, Optional, Sequence, Union

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, Intrinsics
from .labels import Boxes2D, Boxes3D, Masks
from .structures import DataInstance


class LabelInstances(DataInstance):
    """Container for holding ground truth annotations or predictions."""

    def __init__(self, boxes2d: Optional[Boxes2D] = None, boxes3d: Optional[Boxes3D] = None, masks: Optional[Masks] = None) -> None:
        """Init."""
        if boxes2d is None:
            boxes2d = [
                Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))
                for _ in range(len(images))
            ]
        self.boxes2d: Sequence[Boxes2D] = boxes2d

        if boxes3d is None:
            boxes3d = [
                Boxes3D(torch.empty(0, 10), torch.empty(0), torch.empty(0))
                for _ in range(len(images))
            ]
        self.boxes3d: Sequence[Boxes3D] = boxes3d

        if masks is None:
            masks = [
                Masks(torch.empty(0, 1, 1), torch.empty(0), torch.empty(0))
                for i in range(len(images))
            ]
        self.masks = masks

    def to(
            self, device: torch.device
    ) -> "LabelInstances":
        """Move to device (CPU / GPU / ...)."""
        return

    @property
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        if len(self.boxes2d) == 0:
            if len(self.boxes3d) > 0:
                return self.boxes3d.device
            elif len(self.masks) > 0:
                return self.masks.device
        return self.boxes2d.device

    def __getitem__(self, item) -> Instances:
        self.boxes2d = self.boxes2d[mask]
        self.boxes2d = self.boxes2d[mask]
        self.boxes2d = self.boxes2d[mask]


class InputSample(DataInstance):
    """Container holding varying types of DataInstances and Frame metadata."""

    def __init__(
        self,
        metadata: Sequence[Frame],
        images: Images,
        intrinsics: Optional[Intrinsics] = None,
        extrinsics: Optional[Extrinsics] = None,
        targets: LabelInstances = LabelInstances(),
        predictions: LabelInstances = LabelInstances(),
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.images = images
        assert len(metadata) == len(images)
        self.targets = targets
        self.predictions = predictions

        if intrinsics is None:
            intrinsics = Intrinsics(
                torch.cat([torch.eye(3) for _ in range(len(images))])
            )
        self.intrinsics: Intrinsics = intrinsics

        if extrinsics is None:
            extrinsics = Extrinsics(
                torch.cat([torch.eye(4) for _ in range(len(images))])
            )
        self.extrinsics: Extrinsics = extrinsics

    def get(
        self, key: str
    ) -> Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]:
        """Get attribute by key."""
        if key in self.dict():
            value = self.dict()[key]
            return value
        raise AttributeError(f"Attribute {key} not found!")

    def dict(
        self,
    ) -> Dict[
        str, Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]
    ]:
        """Return InputSample object as dict."""
        obj_dict: Dict[
            str, Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]
        ] = {
            "metadata": self.metadata,
            "images": self.images,
            "boxes2d": self.boxes2d,
            "boxes3d": self.boxes3d,
            "masks": self.masks,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
        }
        return obj_dict

    def to(  # pylint: disable=invalid-name
        self, device: torch.device
    ) -> "InputSample":
        """Move to device (CPU / GPU / ...)."""
        return InputSample(
            self.metadata,
            self.images.to(device),
            self.intrinsics.to(device),
            self.extrinsics.to(device),
            self.
        )

    @property
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        return self.images.device

    @classmethod
    def cat(
        cls,
        instances: List["InputSample"],
        device: Optional[torch.device] = None,
    ) -> "InputSample":
        """Concatenate N InputSample objects."""
        cat_dict: Dict[
            str, Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]
        ] = {}
        for k, v in instances[0].dict().items():
            if isinstance(v, list):
                cat_dict[k] = []
                for inst in instances:
                    attr = inst.get(k)
                    cat_dict[k] += [  # type: ignore
                        attr_v.to(device)
                        if isinstance(attr_v, DataInstance)
                        else attr_v
                        for attr_v in attr  # type: ignore
                    ]
            elif isinstance(v, DataInstance) and hasattr(type(v), "cat"):
                cat_dict[k] = type(v).cat(  # type: ignore
                    [inst.get(k) for inst in instances], device
                )
            else:
                raise AttributeError(
                    f"Class {type(v)} for attribute {k} must either be of type"
                    f" list or implement the cat() function (see e.g. Images)!"
                )

        return InputSample(**cat_dict)  # type: ignore

    def __getitem__(self, item: int) -> "InputSample":
        """Return single element."""
        return InputSample(
            [self.metadata[item]],
            self.images[item],
            [self.boxes2d[item]],
            [self.boxes3d[item]],
            [self.masks[item]],
            self.intrinsics[item],
            self.extrinsics[item],
        )

    def __len__(self) -> int:
        """Return number of elements in InputSample."""
        return len(self.metadata)
