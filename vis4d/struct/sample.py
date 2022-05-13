"""Input sample definition in Vis4D."""
from typing import Dict, List, Optional, Tuple, Union

import torch
from scalabel.label.typing import Frame

from .data import Extrinsics, Images, Intrinsics, PointCloud
from .labels import Boxes2D, Boxes3D, InstanceMasks, SemanticMasks
from .structures import DataInstance, InputInstance

# TODO convert to utility functions
# def to(self, device: torch.device) -> "LabelInstances":
#     """Move to device (CPU / GPU / ...)."""
#     return LabelInstances(
#         [b.to(device) for b in self.boxes2d],
#         [b.to(device) for b in self.boxes3d],
#         [m.to(device) for m in self.instance_masks],
#         [m.to(device) for m in self.semantic_masks],
#         [{k: v.to(device) for k, v in o.items()} for o in self.other],
#     )
#
# @property
# def device(self) -> torch.device:
#     """Returns current device if applicable."""
#     return self.boxes2d[0].device
#
# @property
# def empty(self) -> bool:
#     """Returns empty if there are no annotations inside the container."""
#     annotation_sum = 0
#     for i in range(len(self)):
#         annotation_sum += (
#             len(self.boxes2d[i])
#             + len(self.boxes3d[i])
#             + len(self.instance_masks[i])
#             + len(self.semantic_masks[i])
#             + len(self.other[i])
#         )
#     return annotation_sum == 0
#
# def get_instance_labels(
#     self,
# ) -> Tuple[List[Boxes2D], List[Boxes3D], List[InstanceMasks]]:
#     """Get all instance-wise labels."""
#     return self.boxes2d, self.boxes3d, self.instance_masks
#
# def __len__(self) -> int:
#     """Return length of DataInstance."""
#     return len(self.boxes2d)
#
# def __getitem__(self, item: int) -> "LabelInstances":
#     """Return item of DataInstance."""
#     return LabelInstances(
#         [self.boxes2d[item]],
#         [self.boxes3d[item]],
#         [self.instance_masks[item]],
#         [self.semantic_masks[item]],
#     )
#
# @classmethod
# def cat(
#     cls,
#     instances: List["LabelInstances"],
#     device: Optional[torch.device] = None,
# ) -> "LabelInstances":
#     """Concatenate multiple instances into a single one (batching)."""
#     if device is None:
#         device = instances[0].device
#     new_instance = instances[0].to(device)
#     if len(instances) == 1:
#         return new_instance
#     for i in range(1, len(instances)):
#         inst = instances[i].to(device)
#         new_instance.boxes2d.extend(inst.boxes2d)
#         new_instance.boxes3d.extend(inst.boxes3d)
#         new_instance.instance_masks.extend(inst.instance_masks)
#         new_instance.semantic_masks.extend(inst.semantic_masks)
#         new_instance.other.extend(inst.other)
#
#     return new_instance


class Targets(TypedDict):

    boxes2d: Optional[List[torch.Tensor]] = (None,)
    boxes3d: Optional[List[torch.Tensor]] = (None,)
    instance_masks: Optional[List[torch.Tensor]] = (None,)
    semantic_masks: Optional[List[torch.Tensor]] = (None,)
    other: Optional[List[Dict[str, torch.Tensor]]] = (None,)


class MetaData(TypedDict):
    """Input metadata.

    TODO description
    """

    name: str
    url: str
    video_name: Optional[str]
    frame_index: Optional[int]
    timestamp: Optional[torch.Tensor]
    size: Optional[Tuple[int, int]]
    other: Dict[str, torch.Tensor]


class InputData(TypedDict):
    """Container holding the input data.

    TODO description
    """

    metadata: List[MetaData]
    images: Optional[torch.Tensor] = None
    intrinsics: Optional[torch.Tensor] = None
    extrinsics: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None
    targets: Optional[Targets] = None
    other: Optional[List[Dict[str, torch.Tensor]]] = None

    # TODO convert to utility functions
    # def to(  # pylint: disable=invalid-name
    #     self, device: torch.device
    # ) -> "InputSample":
    #     """Move to device (CPU / GPU / ...)."""
    #     return InputSample(
    #         self.metadata,
    #         self.images.to(device),
    #         self.intrinsics.to(device),
    #         self.extrinsics.to(device),
    #         self.points.to(device),
    #         self.targets.to(device),
    #         [{k: v.to(device) for k, v in o.items()} for o in self.other],
    #     )
    #
    # @property
    # def device(self) -> torch.device:
    #     """Returns current device if applicable."""
    #     return self.images.device
    #
    # @classmethod
    # def cat(
    #     cls,
    #     instances: List["InputSample"],
    #     device: Optional[torch.device] = None,
    # ) -> "InputSample":
    #     """Concatenate N InputSample objects."""
    #     cat_dict: Dict[str, InputSampleData] = {}
    #     for k, v in instances[0].dict().items():
    #         if isinstance(v, list):
    #             assert len(v) > 0, "Do not input empty inputSamples to .cat!"
    #             attr_list = []
    #             if isinstance(v[0], dict):
    #                 for inst in instances:
    #                     attr_v = inst.get(k)
    #                     for item in attr_v:
    #                         assert isinstance(item, dict)
    #                         attr_list += [
    #                             {k: v.to(device) for k, v in item.items()}
    #                         ]
    #             else:
    #                 for inst in instances:
    #                     attr_v = inst.get(k)
    #                     assert isinstance(attr_v, list)
    #                     attr_list += attr_v  # type: ignore
    #             cat_dict[k] = attr_list
    #         elif isinstance(v, InputInstance):
    #             cat_dict[k] = type(v).cat(
    #                 [inst.get(k) for inst in instances], device  # type: ignore
    #             )
    #         else:
    #             raise AttributeError(
    #                 f"Class {type(v)} for attribute {k} must be of type list "
    #                 "or InputInstance!"
    #             )
    #
    #     return InputSample(**cat_dict)  # type: ignore
    #
    # def __getitem__(self, item: int) -> "InputSample":
    #     """Return single element."""
    #     return InputSample(
    #         [self.metadata[item]],
    #         self.images[item],
    #         self.intrinsics[item],
    #         self.extrinsics[item],
    #         self.points[item],
    #         self.targets[item],
    #         [self.other[item]],
    #     )
    #
    # def __len__(self) -> int:
    #     """Return number of elements in InputSample."""
    #     return len(self.metadata)
