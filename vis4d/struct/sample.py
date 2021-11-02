"""Input sample definition in Vis4D."""
from typing import Dict, List, Optional, Sequence, Union

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, Intrinsics
from .labels import Boxes2D, Boxes3D, InstanceMasks, SemanticMasks
from .structures import DataInstance


class InputSample(DataInstance):
    """Container holding varying types of DataInstances and Frame metadata."""

    def __init__(
        self,
        metadata: List[Frame],
        images: Images,
        boxes2d: Optional[List[Boxes2D]] = None,
        boxes3d: Optional[List[Boxes3D]] = None,
        instance_masks: Optional[List[InstanceMasks]] = None,
        semantic_masks: Optional[List[SemanticMasks]] = None,
        intrinsics: Optional[Intrinsics] = None,
        extrinsics: Optional[Extrinsics] = None,
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
        self.boxes2d = boxes2d

        if boxes3d is None:
            boxes3d = [
                Boxes3D(torch.empty(0, 10), torch.empty(0), torch.empty(0))
                for _ in range(len(images))
            ]
        self.boxes3d = boxes3d

        if instance_masks is None:
            instance_masks = [
                InstanceMasks(
                    torch.empty(0, 1, 1), torch.empty(0), torch.empty(0)
                )
                for i in range(len(images))
            ]
        self.instance_masks = instance_masks

        if semantic_masks is None:
            semantic_masks = [
                SemanticMasks(
                    torch.empty(0, 1, 1), torch.empty(0), torch.empty(0)
                )
                for i in range(len(images))
            ]
        self.semantic_masks = semantic_masks

        if intrinsics is None:
            intrinsics = Intrinsics.cat(
                [Intrinsics(torch.eye(3)) for _ in range(len(images))]
            )
        self.intrinsics = intrinsics

        if extrinsics is None:
            extrinsics = Extrinsics.cat(
                [Extrinsics(torch.eye(4)) for _ in range(len(images))]
            )
        self.extrinsics = extrinsics

    def get(
        self, key: str
    ) -> Union[List[Frame], DataInstance, List[DataInstance]]:
        """Get attribute by key."""
        if key in self.dict():
            value = self.dict()[key]
            return value
        raise AttributeError(f"Attribute {key} not found!")

    def dict(
        self,
    ) -> Dict[str, Union[List[Frame], DataInstance, List[DataInstance]]]:
        """Return InputSample object as dict."""
        obj_dict: Dict[
            str, Union[List[Frame], DataInstance, List[DataInstance]]
        ] = {
            "metadata": self.metadata,
            "images": self.images,
            "boxes2d": self.boxes2d,  # type: ignore
            "boxes3d": self.boxes3d,  # type: ignore
            "instance_masks": self.instance_masks,  # type: ignore
            "semantic_masks": self.semantic_masks,  # type: ignore
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
            [b.to(device) for b in self.boxes2d],
            [b.to(device) for b in self.boxes3d],
            [m.to(device) for m in self.instance_masks],
            [m.to(device) for m in self.semantic_masks],
            self.intrinsics.to(device),
            self.extrinsics.to(device),
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
                        for attr_v in attr
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
            [self.instance_masks[item]],
            [self.semantic_masks[item]],
            self.intrinsics[item],
            self.extrinsics[item],
        )

    def __len__(self) -> int:
        """Return number of elements in InputSample."""
        return len(self.metadata)
