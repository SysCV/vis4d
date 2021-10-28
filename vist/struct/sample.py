"""Input sample definition in Vist."""
from typing import Dict, List, Optional, Sequence, Union

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, Intrinsics
from .labels import Boxes2D, Boxes3D, Masks
from .structures import DataInstance, TLabelInstance, InputInstance


class LabelInstances(DataInstance):
    """Container for holding ground truth annotations or predictions."""

    def __init__(self, **kwargs: Dict[str, TLabelInstance]) -> None:
        """Init."""
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        # if boxes2d is None:
        #     boxes2d = [
        #         Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))
        #         for _ in range(len(images))
        #     ]
        # self.boxes2d = boxes2d
        #
        # if boxes3d is None:
        #     boxes3d = [
        #         Boxes3D(torch.empty(0, 10), torch.empty(0), torch.empty(0))
        #         for _ in range(len(images))
        #     ]
        # self.boxes3d = boxes3d
        #
        # if masks is None:
        #     masks = [
        #         Masks(torch.empty(0, 1, 1), torch.empty(0), torch.empty(0))
        #         for i in range(len(images))
        #     ]
        # self.masks = masks

    def to(
            self, device: torch.device
    ) -> "LabelInstances":
        """Move to device (CPU / GPU / ...)."""
        # TODO
        return self

    @property
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        # TODO
        return self
        # if len(self.boxes2d) == 0:
        #     if len(self.boxes3d) > 0:
        #         return self.boxes3d.device
        #     elif len(self.masks) > 0:
        #         return self.masks.device
        # return self.boxes2d.device

    def __getitem__(self, item) -> Instances:
        """Get item of LabelInstances."""
        raise NotImplementedError  # TODO


class InputSample(DataInstance):
    """Container holding varying types of DataInstances and Frame metadata."""

    def __init__(
        self,
        metadata: Sequence[Frame],
        images: Images,
        intrinsics: Optional[Intrinsics] = None,
        extrinsics: Optional[Extrinsics] = None,
        targets: List[LabelInstances] = LabelInstances(),
        predictions: List[LabelInstances] = LabelInstances(),
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.images = images
        assert len(metadata) == len(images)
        self.targets = targets
        self.predictions = predictions

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
    ) -> Union[List[Frame], DataInstance]:
        """Get attribute by key."""
        if key in self.dict():
            value = self.dict()[key]
            return value
        raise AttributeError(f"Attribute {key} not found!")

    def dict(
        self,
    ) -> Dict[str, Union[List[Frame], DataInstance]]:
        """Return InputSample object as dict."""
        obj_dict: Dict[
            str, Union[List[Frame], DataInstance]
        ] = {
            "metadata": self.metadata,
            "images": self.images,
            "boxes2d": self.boxes2d,  # type: ignore
            "boxes3d": self.boxes3d,  # type: ignore
            "masks": self.masks,  # type: ignore
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
            self.targets.to(device),
            self.predictions.to(device)
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
                cat_dict[k] = type(v).cat(  # type: ignore
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
        )

    def __len__(self) -> int:
        """Return number of elements in InputSample."""
        return len(self.metadata)
