"""Data structure for structures container."""
import abc
import copy
from typing import Dict, List, Tuple

import torch
from scalabel.label.typing import Box2D, Label


class Instances(metaclass=abc.ABCMeta):
    """Interface for bounding boxes, masks etc"""

    @classmethod
    @abc.abstractmethod
    def from_scalabel(
        cls, labels: List[Label], class_to_idx: Dict[str, int]
    ) -> "Instances":
        """Convert from scalabel format to ours."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_scalabel(cls, idx_to_class: Dict[int, str]) -> List[Label]:
        """Convert from ours to scalabel format."""
        raise NotImplementedError


class Boxes2D(Instances):
    """Container class for 2D boxes.
    boxes: torch.FloatTensor: (N, 5) where each entry is defined by
    [x1, y1, x2, y2, score]
    classes: torch.IntTensor: (N,) where each entry is the class id of the respective box
    track_ids: torch.IntTensor (N,) where each entry is the track id of the respective box
    """

    def __init__(
        self,
        boxes: torch.Tensor,
        classes: torch.Tensor = None,
        track_ids: torch.Tensor = None,
        image_wh: Tuple[int, int] = None,
    ) -> None:
        """Init."""  # TODO image wh needed?
        assert isinstance(boxes, torch.Tensor) and len(boxes.shape) == 2
        if classes is not None:
            assert (
                isinstance(classes, torch.Tensor)
                and len(boxes) == len(classes)
                and boxes.device == classes.device
            )
        if track_ids is not None:
            assert (
                isinstance(track_ids, torch.Tensor)
                and len(boxes) == len(track_ids)
                and boxes.device == track_ids.device
            )
        self.boxes = boxes
        self.classes = classes
        self.track_ids = track_ids
        self.image_wh = image_wh

    def __getitem__(self, item):
        """This method will shadow the tensor based indexing while returning a
        new instance of Boxes2D."""
        boxes = self.boxes[item]
        classes = self.classes[item] if self.classes is not None else None
        track_ids = (
            self.track_ids[item] if self.track_ids is not None else None
        )
        if isinstance(item, int):
            if classes is not None:
                classes = classes.view(1, -1)
            if track_ids is not None:
                track_ids = track_ids.view(1, -1)
            return Boxes2D(
                boxes.view(1, -1), classes, track_ids, self.image_wh
            )
        else:
            return Boxes2D(boxes, classes, track_ids, self.image_wh)

    def __len__(self):
        """Get length of the object"""
        return len(self.boxes)

    def clone(self) -> "Boxes2D":
        """Create a copy of the object."""
        classes = self.classes.clone() if self.classes is not None else None
        track_ids = (
            self.track_ids.clone() if self.track_ids is not None else None
        )
        return Boxes2D(self.boxes.clone(), classes, track_ids, self.image_wh)

    def to(self, device: torch.device):
        """Move data to given device."""
        classes = (
            self.classes.to(device=device)
            if self.classes is not None
            else None
        )
        track_ids = (
            self.track_ids.to(device=device)
            if self.track_ids is not None
            else None
        )
        return Boxes2D(
            self.boxes.to(device=device), classes, track_ids, self.image_wh
        )

    @property
    def device(self) -> torch.device:
        """Get current device of data."""
        return self.boxes.device

    @classmethod
    def from_scalabel(
        cls, labels: List[Label], class_to_idx: Dict[str, int]
    ) -> "Boxes2D":
        """Convert from scalabel format to ours."""  # TODO image wh
        box_list, cls_list = [], []
        for label in labels:
            box, score, box_cls = (
                label.box_2d,
                label.score,
                label.attributes["category"],
            )
            box_list.append([box.x1, box.y1, box.x2, box.y2, score])
            cls_list.append(class_to_idx[box_cls])

        return Boxes2D(
            torch.tensor(box_list, dtype=torch.float32),
            torch.tensor(cls_list, dtype=torch.int),
            torch.tensor(cls_list, dtype=torch.int),
        )

    def to_scalabel(self, idx_to_class: Dict[int, str]) -> List[Label]:
        """Convert from ours to scalabel format."""  # TODO image wh
        labels = []
        for i in range(len(self.boxes)):
            box = Box2D(
                x1=float(self.boxes[i, 0]),
                y1=float(self.boxes[i, 1]),
                x2=float(self.boxes[i, 2]),
                y2=float(self.boxes[i, 3]),
            )
            score = float(self.boxes[i, 4])
            cls = idx_to_class[int(self.classes[i])]
            attributes = dict(category=cls)
            attributes["id"] = str(self.track_ids[i].item())
            labels.append(
                Label(box_2d=box, score=score, attributes=attributes)
            )

        return labels
