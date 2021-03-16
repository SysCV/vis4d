"""Data structure for label container."""
import abc
from typing import Dict, List, Tuple

import torch
from scalabel.label.typing import Box2D, Label

# adapt typing




class Instances(metaclass=abc.ABCMeta):
    """Interface for bounding boxes, masks etc"""

    @abc.abstractmethod
    def from_scalabel(self, labels: List[Label], class_to_idx: Dict[str, int]):
        """Convert from scalabel format to ours."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_scalabel(self, idx_to_class: Dict[int, str]):
        """Convert from ours to scalabel format."""
        raise NotImplementedError


class Boxes2D(Instances):
    """Container class for 2D boxes.
    data: torch.FloatTensor: Nx6 where each entry is defined by
    [x1, y1, x2, y2, score, class index]

    """

    def __init__(self, data: torch.tensor = None) -> None:
        """Init."""
        self.data = data

    def from_scalabel(self, labels: List[Label], class_to_idx: Dict[str, int]):
        """Convert from scalabel format to ours."""
        box_list = []
        for label in labels:
            box, score, cls = (
                label.box_2d,
                label.score,
                label.attributes["category"],
            )
            box_list.append(
                [box.x1, box.y1, box.x2, box.y2, score, class_to_idx[cls]]
            )

        self.data = torch.tensor(box_list, dtype=torch.float32)

    def to_scalabel(self, idx_to_class: Dict[int, str]) -> List[Label]:
        """Convert from ours to scalabel format."""
        labels = []
        for box_data in self.data:
            box = Box2D(
                x1=float(box_data[0]),
                y1=float(box_data[1]),
                x2=float(box_data[2]),
                y2=float(box_data[3]),
            )
            score = float(box_data[4])
            cls = idx_to_class[int(box_data[5])]
            labels.append(
                Label(box_2d=box, score=score, attributes=dict(category=cls))
            )

        return labels
