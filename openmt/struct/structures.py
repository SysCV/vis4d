"""Data structure for struct container."""
import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from detectron2.structures import ImageList as D2ImageList
from scalabel.label.typing import Box2D, Label

ImageList = D2ImageList
TorchCheckpoint = Dict[str, Union[int, str, Dict[str, np.ndarray]]]


class Instances(metaclass=abc.ABCMeta):
    """Interface for bounding boxes, masks etc."""

    @classmethod
    @abc.abstractmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
    ) -> "Instances":
        """Convert from scalabel format to ours."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_scalabel(self, idx_to_class: Dict[int, str]) -> List[Label]:
        """Convert from ours to scalabel format."""
        raise NotImplementedError


class Boxes2D(Instances):
    """Container class for 2D boxes.

    boxes: torch.FloatTensor: (N, 5) where each entry is defined by
    [x1, y1, x2, y2, score]
    class_ids: torch.IntTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.IntTensor (N,) where each entry is the track id of
    the respective box.
    """

    def __init__(
        self,
        boxes: torch.Tensor,
        class_ids: torch.Tensor = None,
        track_ids: torch.Tensor = None,
        image_wh: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Init."""
        assert isinstance(boxes, torch.Tensor) and len(boxes.shape) == 2
        if class_ids is not None:
            assert (
                isinstance(class_ids, torch.Tensor)
                and len(boxes) == len(class_ids)
                and boxes.device == class_ids.device
            )
        if track_ids is not None:
            assert (
                isinstance(track_ids, torch.Tensor)
                and len(boxes) == len(track_ids)
                and boxes.device == track_ids.device
            )
        self.boxes = boxes
        self.class_ids = class_ids
        self.track_ids = track_ids
        self.image_wh = image_wh

    def __getitem__(self, item) -> "Boxes2D":  # type: ignore
        """Shadows tensor based indexing while returning new Boxes2D."""
        if isinstance(item, tuple):
            item = item[0]
        boxes = self.boxes[item]
        class_ids = (
            self.class_ids[item] if self.class_ids is not None else None
        )
        track_ids = (
            self.track_ids[item] if self.track_ids is not None else None
        )
        if len(boxes.shape) < 2:
            if class_ids is not None:
                class_ids = class_ids.view(1, -1)
            if track_ids is not None:
                track_ids = track_ids.view(1, -1)
            return Boxes2D(
                boxes.view(1, -1), class_ids, track_ids, self.image_wh
            )

        return Boxes2D(boxes, class_ids, track_ids, self.image_wh)

    def __len__(self) -> int:
        """Get length of the object."""
        return len(self.boxes)

    def clone(self) -> "Boxes2D":
        """Create a copy of the object."""
        class_ids = (
            self.class_ids.clone() if self.class_ids is not None else None
        )
        track_ids = (
            self.track_ids.clone() if self.track_ids is not None else None
        )
        return Boxes2D(self.boxes.clone(), class_ids, track_ids, self.image_wh)

    def to(  # pylint: disable=invalid-name
        self, device: torch.device
    ) -> "Boxes2D":
        """Move data to given device."""
        class_ids = (
            self.class_ids.to(device=device)
            if self.class_ids is not None
            else None
        )
        track_ids = (
            self.track_ids.to(device=device)
            if self.track_ids is not None
            else None
        )
        return Boxes2D(
            self.boxes.to(device=device), class_ids, track_ids, self.image_wh
        )

    @classmethod
    def cat(cls, boxes_list: List["Boxes2D"]) -> "Boxes2D":
        """Concatenates a list of Boxes2D into a single Boxes2D."""
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all((isinstance(box, Boxes2D) for box in boxes_list))

        boxes, class_ids, track_ids = [], [], []
        has_class_ids = all((b.class_ids is not None for b in boxes_list))
        has_track_ids = all((b.track_ids is not None for b in boxes_list))
        for b in boxes_list:
            boxes.append(b.boxes)
            if has_class_ids:
                class_ids.append(b.class_ids)
            if has_track_ids:
                track_ids.append(b.track_ids)

        cat_boxes = cls(
            torch.cat(boxes),
            torch.cat(class_ids) if has_class_ids else None,
            torch.cat(track_ids) if has_track_ids else None,
        )
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
        label_id_to_idx: Optional[Dict[str, int]] = None,
    ) -> "Boxes2D":
        """Convert from scalabel format to internal."""
        box_list, cls_list, idx_list = [], [], []
        has_class_ids = all((b.category is not None for b in labels))
        for i, label in enumerate(labels):
            box, score, box_cls, l_id = (
                label.box_2d,
                label.score,
                label.category,
                label.id,
            )
            if box is None:
                continue

            box_list.append([box.x1, box.y1, box.x2, box.y2, score])
            if has_class_ids:
                cls_list.append(class_to_idx[box_cls])  # type: ignore
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)

        return Boxes2D(
            torch.tensor(box_list, dtype=torch.float32),
            torch.tensor(cls_list, dtype=torch.int) if has_class_ids else None,
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

            cls = idx_to_class[int(self.class_ids[i])]
            label_dict["category"] = cls
            labels.append(Label(**label_dict))

        return labels
