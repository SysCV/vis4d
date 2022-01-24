"""Input sample definition in Vis4D."""
from typing import Dict, List, Optional, Tuple, Union

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, Intrinsics, PointCloud
from .labels import Boxes2D, Boxes3D, InstanceMasks, SemanticMasks
from .structures import DataInstance, InputInstance

InputSampleData = Union[
    List[Frame], DataInstance, List[Dict[str, torch.Tensor]]
]


class LabelInstances(InputInstance):
    """Container for ground truth annotations / predictions."""

    def __init__(
        self,
        boxes2d: Optional[List[Boxes2D]] = None,
        boxes3d: Optional[List[Boxes3D]] = None,
        instance_masks: Optional[List[InstanceMasks]] = None,
        semantic_masks: Optional[List[SemanticMasks]] = None,
        other: Optional[List[Dict[str, torch.Tensor]]] = None,
        default_len: int = 1,
    ) -> None:
        """Init."""
        inputs = (boxes2d, boxes3d, instance_masks, semantic_masks, other)
        annotation_len = default_len
        device = torch.device("cpu")
        if not all(x is None for x in inputs):
            for x in inputs:
                if x is not None:
                    annotation_len = len(x)
                    if isinstance(x[0], dict):
                        device = x[0][list(x[0].keys())[0]].device
                    else:
                        device = x[0].device
                    break

        if boxes2d is None:
            boxes2d = [Boxes2D.empty(device) for _ in range(annotation_len)]
        self.boxes2d = boxes2d

        if boxes3d is None:
            boxes3d = [Boxes3D.empty(device) for _ in range(annotation_len)]
        self.boxes3d = boxes3d

        if instance_masks is None:
            instance_masks = [
                InstanceMasks.empty(device) for _ in range(annotation_len)
            ]
        self.instance_masks = instance_masks

        if semantic_masks is None:
            semantic_masks = [
                SemanticMasks.empty(device) for _ in range(annotation_len)
            ]
        self.semantic_masks = semantic_masks

        if other is None:
            other = [{} for _ in range(annotation_len)]
        self.other = other

    def to(self, device: torch.device) -> "LabelInstances":
        """Move to device (CPU / GPU / ...)."""
        return LabelInstances(
            [b.to(device) for b in self.boxes2d],
            [b.to(device) for b in self.boxes3d],
            [m.to(device) for m in self.instance_masks],
            [m.to(device) for m in self.semantic_masks],
            [{k: v.to(device) for k, v in o.items()} for o in self.other],
        )

    @property
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        return self.boxes2d[0].device

    @property
    def empty(self) -> bool:
        """Returns empty if there are no annotations inside the container."""
        annotation_sum = 0
        for i in range(len(self)):
            annotation_sum += (
                len(self.boxes2d[i])
                + len(self.boxes3d[i])
                + len(self.instance_masks[i])
                + len(self.semantic_masks[i])
                + len(self.other[i])
            )
        return annotation_sum == 0

    def get_instance_labels(
        self,
    ) -> Tuple[List[Boxes2D], List[Boxes3D], List[InstanceMasks]]:
        """Get all instance-wise labels."""
        return self.boxes2d, self.boxes3d, self.instance_masks

    def __len__(self) -> int:
        """Return length of DataInstance."""
        return len(self.boxes2d)

    def __getitem__(self, item: int) -> "LabelInstances":
        """Return item of DataInstance."""
        return LabelInstances(
            [self.boxes2d[item]],
            [self.boxes3d[item]],
            [self.instance_masks[item]],
            [self.semantic_masks[item]],
        )

    @classmethod
    def cat(
        cls,
        instances: List["LabelInstances"],
        device: Optional[torch.device] = None,
    ) -> "LabelInstances":
        """Concatenate multiple instances into a single one (batching)."""
        if device is None:
            device = instances[0].device
        new_instance = instances[0].to(device)
        if len(instances) == 1:
            return new_instance
        for i in range(1, len(instances)):
            inst = instances[i].to(device)
            new_instance.boxes2d.extend(inst.boxes2d)
            new_instance.boxes3d.extend(inst.boxes3d)
            new_instance.instance_masks.extend(inst.instance_masks)
            new_instance.semantic_masks.extend(inst.semantic_masks)
            new_instance.other.extend(inst.other)

        return new_instance


class InputSample(DataInstance):
    """Container holding varying types of DataInstances and Frame metadata."""

    def __init__(
        self,
        metadata: List[Frame],
        images: Images,
        intrinsics: Optional[Intrinsics] = None,
        extrinsics: Optional[Extrinsics] = None,
        points: Optional[PointCloud] = None,
        targets: Optional[LabelInstances] = None,
        other: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.images = images
        assert len(metadata) == len(images)

        if intrinsics is None:
            intrinsics = Intrinsics.cat(
                [Intrinsics(torch.eye(3)) for _ in range(len(metadata))]
            )
        self.intrinsics = intrinsics

        if extrinsics is None:
            extrinsics = Extrinsics.cat(
                [Extrinsics(torch.eye(4)) for _ in range(len(metadata))]
            )
        self.extrinsics = extrinsics

        if points is None:
            points = PointCloud(torch.cat([torch.empty(len(images), 1, 4)]))
        self.points = points

        if targets is None:
            targets = LabelInstances(default_len=len(metadata))
        self.targets = targets

        if other is None:
            other = [{} for _ in range(len(metadata))]
        self.other = other

    def get(self, key: str) -> InputSampleData:
        """Get attribute by key."""
        if key in self.dict():
            value = self.dict()[key]
            return value
        raise AttributeError(f"Attribute {key} not found!")

    def dict(self) -> Dict[str, InputSampleData]:
        """Return InputSample object as dict."""
        obj_dict: Dict[str, InputSampleData] = {
            "metadata": self.metadata,
            "images": self.images,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "points": self.points,
            "targets": self.targets,
            "other": self.other,
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
            self.points.to(device),
            self.targets.to(device),
            [{k: v.to(device) for k, v in o.items()} for o in self.other],
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
        cat_dict: Dict[str, InputSampleData] = {}
        for k, v in instances[0].dict().items():
            if isinstance(v, list):
                assert len(v) > 0, "Do not input empty inputSamples to .cat!"
                attr_list = []
                if isinstance(v[0], dict):
                    for inst in instances:
                        attr_v = inst.get(k)
                        for item in attr_v:
                            assert isinstance(item, dict)
                            attr_list += [
                                {k: v.to(device) for k, v in item.items()}
                            ]
                else:
                    for inst in instances:
                        attr_v = inst.get(k)
                        assert isinstance(attr_v, list)
                        attr_list += attr_v  # type: ignore
                cat_dict[k] = attr_list
            elif isinstance(v, InputInstance):
                cat_dict[k] = type(v).cat(
                    [inst.get(k) for inst in instances], device  # type: ignore
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
            self.points[item],
            self.targets[item],
            [self.other[item]],
        )

    def __len__(self) -> int:
        """Return number of elements in InputSample."""
        return len(self.metadata)
