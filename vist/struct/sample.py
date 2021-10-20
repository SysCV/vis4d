"""Input sample definition in Vist."""
from typing import Dict, List, Optional, Sequence, Union

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, Intrinsics
from .labels import Boxes2D, Boxes3D
from .structures import DataInstance, NDArrayF32


class InputSample:
    """Container holding varying types of DataInstances and Frame metadata."""

    def __init__(
        self,
        metadata: Sequence[Frame],
        images: Images,
        boxes2d: Optional[Sequence[Boxes2D]] = None,
        boxes3d: Optional[Sequence[Boxes3D]] = None,
        intrinsics: Optional[Intrinsics] = None,
        extrinsics: Optional[Extrinsics] = None,
        points: Optional[Sequence[NDArrayF32]] = None,
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.images = images
        assert len(metadata) == len(images)

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

        if points is None:
            points = [torch.empty(0, 4) for _ in range(len(images))]
        self.points = [points]

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
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "points": self.points,
        }
        return obj_dict

    def to(  # pylint: disable=invalid-name
        self, device: torch.device
    ) -> "InputSample":
        """Move to device (CPU / GPU / ...)."""
        return InputSample(
            self.metadata,
            self.images.to(device),
            [b.to(device) for b in self.boxes2d],
            [b.to(device) for b in self.boxes3d],
            self.intrinsics.to(device),
            self.extrinsics.to(device),
            self.points,
        )

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
            self.intrinsics[item],
            self.extrinsics[item],
            [self.points[item]],
        )
