"""Data structure for struct container."""
import abc
from typing import Dict, List, Tuple

import torch
from scalabel.label.typing import Box2D, Label


class Instances(metaclass=abc.ABCMeta):
    """Interface for bounding boxes, masks etc."""

    @classmethod
    @abc.abstractmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Dict[str, int] = None,
    ) -> "Instances":
        """Convert from scalabel format to ours."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def to_scalabel(cls, idx_to_class: Dict[int, str]) -> List[Label]:
        """Convert from ours to scalabel format."""
        raise NotImplementedError


class Boxes2D(Instances):
    """Container class for 2D boxes.

    boxes: torch.FloatTensor: (N, 5) where each entry is defined by
    [x1, y1, x2, y2, score]
    classes: torch.IntTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.IntTensor (N,) where each entry is the track id of
    the respective box.
    """

    def __init__(
        self,
        boxes: torch.Tensor,
        classes: torch.Tensor = None,
        track_ids: torch.Tensor = None,
        image_wh: Tuple[int, int] = None,
    ) -> None:
        """Init."""
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
        """Shadows tensor based indexing while returning new Boxes2D."""
        if isinstance(item, tuple):
            item = item[0]
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

        return Boxes2D(boxes, classes, track_ids, self.image_wh)

    def __len__(self):
        """Get length of the object."""
        return len(self.boxes)

    def clone(self) -> "Boxes2D":
        """Create a copy of the object."""
        classes = self.classes.clone() if self.classes is not None else None
        track_ids = (
            self.track_ids.clone() if self.track_ids is not None else None
        )
        return Boxes2D(self.boxes.clone(), classes, track_ids, self.image_wh)

    def to(self, device: torch.device):  # pylint: disable=invalid-name
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

    @classmethod
    def cat(cls, boxes_list: List["Boxes2D"]) -> "Boxes2D":
        """Concatenates a list of Boxes2D into a single Boxes2D."""
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all((isinstance(box, Boxes2D) for box in boxes_list))

        boxes, classes, track_ids = [], [], []
        for b in boxes_list:
            boxes.append(b.boxes)
            if classes is not None:
                if b.classes is not None:
                    classes.append(b.classes)
                else:
                    classes = None

                if b.classes is not None:
                    track_ids.append(b.track_ids)
                else:
                    track_ids = None
        boxes = torch.cat(boxes)
        classes = torch.cat(classes) if classes is not None else None
        track_ids = torch.cat(track_ids) if track_ids is not None else None
        cat_boxes = cls(boxes, classes, track_ids)
        return cat_boxes

    @property
    def device(self) -> torch.device:
        """Get current device of data."""
        return self.boxes.device

    @classmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Dict[str, int] = None,
    ) -> "Boxes2D":
        """Convert from scalabel format to internal."""
        box_list, cls_list, idx_list = [], [], []
        for i, label in enumerate(labels):
            box, score, box_cls, l_id = (
                label.box_2d,
                label.score,
                label.category,
                label.id,
            )
            box_list.append([box.x1, box.y1, box.x2, box.y2, score])
            cls_list.append(class_to_idx[box_cls])
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)

        return Boxes2D(
            torch.tensor(box_list, dtype=torch.float32),
            torch.tensor(cls_list, dtype=torch.int),
            torch.tensor(idx_list, dtype=torch.int),
        )

    def to_scalabel(self, idx_to_class: Dict[int, str]) -> List[Label]:
        """Convert from internal to scalabel format."""
        labels = []
        for i in range(len(self.boxes)):
            if self.track_ids is not None:
                label_id = str(self.track_ids[i].item())
            else:
                label_id = str(i)
            box = Box2D(
                x1=float(self.boxes[i, 0]),
                y1=float(self.boxes[i, 1]),
                x2=float(self.boxes[i, 2]),
                y2=float(self.boxes[i, 3]),
            )

            score = float(self.boxes[i, 4])
            label_dict = dict(id=label_id, box_2d=box, score=score)

            cls = idx_to_class[int(self.classes[i])]
            label_dict["category"] = cls
            labels.append(Label(**label_dict))

        return labels
