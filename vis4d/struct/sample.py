"""Input sample definition in Vis4D."""
from typing import Dict, List, Optional, Sequence, Union

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, Intrinsics
from .labels import Boxes2D, Boxes3D, InstanceMasks, SemanticMasks
from .structures import DataInstance, InputInstance, TLabelInstance


class Targets(InputInstance):
    """Container for all ground truth annotations."""

    def __init__(
        self,
        boxes2d: Optional[List[Boxes2D]] = None,
        boxes3d: Optional[List[Boxes3D]] = None,
        instance_masks: Optional[List[InstanceMasks]] = None,
        semantic_masks: Optional[List[SemanticMasks]] = None,
    ) -> None:
        """Init."""
        assert any([]), "Container should not be empty"
        annotation_len = []

        if boxes2d is None:
            boxes2d = [
                Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))
                for _ in range()
            ]
        self.boxes2d = boxes2d

        if boxes3d is None:
            boxes3d = Boxes3D(
                torch.empty(0, 10), torch.empty(0), torch.empty(0)
            )
        self.boxes3d = boxes3d

        if instance_masks is None:
            instance_masks = InstanceMasks(
                torch.empty(0, 1, 1), torch.empty(0), torch.empty(0)
            )
        self.instance_masks = instance_masks

        if semantic_masks is None:
            semantic_masks = SemanticMasks(
                torch.empty(0, 1, 1), torch.empty(0), torch.empty(0)
            )
        self.semantic_masks = semantic_masks

    @abc.abstractmethod
    def to(  # pylint: disable=invalid-name
        self: "TDataInstance", device: torch.device
    ) -> "TDataInstance":
        """Move to device (CPU / GPU / ...)."""
        return Targets(
            self.boxes2d.to(device),
            self.boxes3d.to(device),
            self.instance_masks.to(device),
            self.semantic_masks.to(device),
        )

    @property
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return length of DataInstance."""
        raise NotImplementedError

    def __getitem__(self: "TDataInstance", item: int) -> "TDataInstance":
        """Return item of DataInstance."""
        raise NotImplementedError


class InputSample(DataInstance):
    """Container holding varying types of DataInstances and Frame metadata."""

    def __init__(
        self,
        metadata: Sequence[Frame],
        images: Images,
        intrinsics: Optional[Intrinsics] = None,
        extrinsics: Optional[Extrinsics] = None,
        targets: Optional[Targets] = None,
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.images = images
        assert len(metadata) == len(images)

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

        if targets is None:
            targets = Targets()
        self.targets = targets

    def get(self, key: str) -> Union[List[Frame], DataInstance]:
        """Get attribute by key."""
        if key in self.dict():
            value = self.dict()[key]
            return value
        raise AttributeError(f"Attribute {key} not found!")

    def dict(
        self,
    ) -> Dict[str, Union[List[Frame], DataInstance]]:
        """Return InputSample object as dict."""
        obj_dict: Dict[str, Union[List[Frame], DataInstance]] = {
            "metadata": self.metadata,
            "images": self.images,
            "targets": self.targets,
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
            [t.to(device) for t in self.targets],
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
        cat_dict: Dict[str, Union[Sequence[Frame], DataInstance]] = {}
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
            elif isinstance(v, InputInstance):
                cat_dict[k] = type(v).cat(
                    [inst.get(k) for inst in instances], device
                )
            else:
                raise AttributeError(
                    f"Class {type(v)} for attribute {k} must be of type list "
                    "or InputInstance!"
                )

        return InputSample(**cat_dict)  # type: ignore

    def __getitem__(self, item: int) -> "InputSample":
        """Return single element."""
        return InputSample(
            [self.metadata[item]],
            self.images[item],
            self.intrinsics[item],
            self.extrinsics[item],
            self.targets[item],
        )

    def __len__(self) -> int:
        """Return number of elements in InputSample."""
        return len(self.metadata)
