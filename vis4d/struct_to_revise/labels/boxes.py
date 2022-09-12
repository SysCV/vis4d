"""Vis4D Boxes data structures."""
import abc
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import torch
from scalabel.label.transforms import box2d_to_xyxy, xyxy_to_box2d
from scalabel.label.typing import Box3D, ImageSize, Label

from vis4d.common_to_revise.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)

from ..data import Extrinsics
from ..structures import LabelInstance

TBoxes = TypeVar("TBoxes", bound="Boxes")


class Boxes(LabelInstance):
    """Abstract container for 2D / BEV / 3D / ... Boxes.

    boxes: torch.FloatTensor: (N, M) N elements of boxes with M parameters
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
    ) -> None:
        """Init."""
        assert isinstance(boxes, torch.Tensor) and len(boxes.shape) == 2
        if class_ids is not None:
            assert isinstance(class_ids, torch.Tensor)
            assert len(boxes) == len(class_ids)
            assert boxes.device == class_ids.device
        if track_ids is not None:
            assert isinstance(track_ids, torch.Tensor)
            assert len(boxes) == len(track_ids)
            assert boxes.device == track_ids.device

        self.boxes = boxes
        self.class_ids = class_ids
        self.track_ids = track_ids

    def __getitem__(self: "TBoxes", item) -> "TBoxes":  # type: ignore
        """Shadows tensor based indexing while returning new Boxes."""
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
            return type(self)(boxes.view(1, -1), class_ids, track_ids)

        return type(self)(boxes, class_ids, track_ids)

    def __len__(self) -> int:
        """Get length of the object."""
        return len(self.boxes)

    def clone(self: "TBoxes") -> "TBoxes":
        """Create a copy of the object."""
        class_ids = (
            self.class_ids.clone() if self.class_ids is not None else None
        )
        track_ids = (
            self.track_ids.clone() if self.track_ids is not None else None
        )
        return type(self)(self.boxes.clone(), class_ids, track_ids)

    def to(self: "TBoxes", device: torch.device) -> "TBoxes":
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
        return type(self)(self.boxes.to(device=device), class_ids, track_ids)

    @classmethod
    def merge(cls: Type["TBoxes"], instances: List["TBoxes"]) -> "TBoxes":
        """Merges a list of Boxes into a single Boxes.

        If the Boxes instances have different number of parameters per entry,
        this function will take the minimum and cut additional parameters
        from the other instances.
        """
        assert isinstance(instances, (list, tuple))
        assert len(instances) > 0
        assert all((isinstance(inst, Boxes) for inst in instances))
        assert all(instances[0].device == inst.device for inst in instances)

        boxes, class_ids, track_ids = [], [], []
        has_class_ids = all((b.class_ids is not None for b in instances))
        has_track_ids = all((b.track_ids is not None for b in instances))
        min_param = min((b.boxes.shape[-1] for b in instances))
        for b in instances:
            boxes.append(b.boxes[:, :min_param])
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
    def empty(
        cls: Type["TBoxes"], device: Optional[torch.device] = None
    ) -> "TBoxes":
        """Return empty boxes on device."""
        return cls(torch.empty(0, 5), torch.empty(0), torch.empty(0)).to(
            device
        )

    @classmethod
    @abc.abstractmethod
    def from_scalabel(
        cls: Type["TBoxes"],
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> "TBoxes":
        """Convert from scalabel format to ours."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> List[Label]:
        """Convert from ours to scalabel format."""
        raise NotImplementedError


class Boxes2D(Boxes):
    """Container class for 2D boxes.

    boxes: torch.FloatTensor: (N, [4, 5]) where each entry is defined by
    [x1, y1, x2, y2, Optional[score]]
    class_ids: torch.LongTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.LongTensor (N,) where each entry is the track id of
    the respective box.
    """

    def scale(self, scale_factor_xy: Tuple[float, float]) -> None:
        """Scale bounding boxes according to factor."""
        self.boxes[:, [0, 2]] *= scale_factor_xy[0]
        self.boxes[:, [1, 3]] *= scale_factor_xy[1]

    def clip(self, image_wh: Tuple[float, float]) -> None:
        """Clip bounding boxes according to image_wh."""
        self.boxes[:, [0, 2]] = self.boxes[:, [0, 2]].clamp(0, image_wh[0] - 1)
        self.boxes[:, [1, 3]] = self.boxes[:, [1, 3]].clamp(0, image_wh[1] - 1)

    @property
    def score(self) -> Optional[torch.Tensor]:
        """Return scores of 2D bounding boxes as tensor."""
        if not self.boxes.shape[-1] == 5:
            return None
        return self.boxes[:, -1]

    @property
    def center(self) -> torch.Tensor:
        """Return center of 2D bounding boxes as tensor."""
        ctr_x = (self.boxes[:, 0] + self.boxes[:, 2]) / 2
        ctr_y = (self.boxes[:, 1] + self.boxes[:, 3]) / 2
        return torch.stack([ctr_x, ctr_y], -1)

    @property
    def area(self) -> torch.Tensor:
        """Compute area of each bounding box."""
        area = (self.boxes[:, 2] - self.boxes[:, 0]).clamp(0) * (
            self.boxes[:, 3] - self.boxes[:, 1]
        ).clamp(0)
        return area

    @classmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> "Boxes2D":
        """Convert from scalabel format to internal.

        NOTE: The box definition in Scalabel includes x2y2, whereas Vis4D and
        other software libraries like detectron2, mmdet do not include this,
        which is why we convert via box2d_to_xyxy.
        """
        box_list, cls_list, idx_list = [], [], []
        has_class_ids = all((b.category is not None for b in labels))
        for i, label in enumerate(labels):
            box, score, box_cls, l_id = (
                label.box2d,
                label.score,
                label.category,
                label.id,
            )
            if box is None:
                continue
            if has_class_ids:
                if box_cls in class_to_idx:
                    cls_list.append(class_to_idx[box_cls])
                else:  # pragma: no cover
                    continue

            if score is None:
                box_list.append([*box2d_to_xyxy(box)])
            else:
                box_list.append([*box2d_to_xyxy(box), score])

            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)

        if len(box_list) == 0:  # pragma: no cover
            return cls.empty()
        box_tensor = torch.tensor(box_list, dtype=torch.float32)
        class_ids = (
            torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
        )
        track_ids = torch.tensor(idx_list, dtype=torch.long)
        return Boxes2D(box_tensor, class_ids, track_ids)

    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> List[Label]:
        """Convert from internal to scalabel format."""
        labels = []
        for i in range(len(self.boxes)):
            if self.track_ids is not None:
                label_id = str(self.track_ids[i].item())
            else:
                label_id = str(i)
            box = xyxy_to_box2d(
                float(self.boxes[i, 0]),
                float(self.boxes[i, 1]),
                float(self.boxes[i, 2]),
                float(self.boxes[i, 3]),
            )
            if self.boxes.shape[-1] == 5:
                score: Optional[float] = float(self.boxes[i, 4])
            else:
                score = None
            label_dict = dict(id=label_id, box2d=box, score=score)

            if idx_to_class is not None:
                cls = idx_to_class[int(self.class_ids[i])]
            else:
                cls = str(int(self.class_ids[i]))  # pragma: no cover
            label_dict["category"] = cls
            labels.append(Label(**label_dict))

        return labels

    def postprocess(
        self,
        original_wh: Tuple[int, int],
        output_wh: Tuple[int, int],
        clip: bool = True,
        resolve_overlap: bool = True,
    ) -> None:
        """Postprocess boxes."""
        scale_factor = (
            original_wh[0] / output_wh[0],
            original_wh[1] / output_wh[1],
        )
        self.scale(scale_factor)
        if clip:
            self.clip(original_wh)


def tensor_to_boxes2d(
    boxes: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    class_ids: Optional[torch.Tensor] = None,
    track_ids: Optional[torch.Tensor] = None,
) -> Boxes2D:
    """Convert Tensors to Boxes2D."""
    if scores is not None:
        boxes_ = torch.cat([boxes, scores], -1)
    else:
        boxes_ = boxes
    return Boxes2D(boxes_, class_ids, track_ids)


def boxes2d_to_tensor(
    boxes2d: Boxes2D,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Convert Tensors to Boxes2D."""
    scores = boxes2d.score
    return boxes2d.boxes[:, :4], scores, boxes2d.class_ids, boxes2d.track_ids


def tensor_list_to_boxes2d_list(
    boxes: List[torch.Tensor],
    scores: Optional[List[torch.Tensor]] = None,
    class_ids: Optional[List[torch.Tensor]] = None,
    track_ids: Optional[List[torch.Tensor]] = None,
) -> List[Boxes2D]:
    boxes2d_list = []
    for i in range(len(boxes)):
        score = scores[i] if scores is not None else None
        cls_ids = class_ids[i] if class_ids is not None else None
        tr_ids = track_ids[i] if track_ids is not None else None
        boxes2d_list.append(
            tensor_to_boxes2d(boxes[i], score, cls_ids, tr_ids)
        )
    return boxes2d_list


def boxes2d_list_to_tensor_list(
    boxes2d: List[Boxes2D],
) -> Tuple[
    List[torch.Tensor],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
]:
    boxes_list, scores_list, cls_list, tr_list = [], [], [], []
    for boxs2d in boxes2d:
        boxes, scores, cls_ids, tr_ids = boxes2d_to_tensor(boxs2d)
        boxes_list.append(boxes)
        if scores is not None:
            scores_list.append(scores)
        if cls_ids is not None:
            cls_list.append(cls_ids)
        if tr_ids is not None:
            tr_list.append(tr_ids)

        assert len(scores_list) == 0 or len(scores_list) == len(boxes_list)
        assert len(cls_list) == 0 or len(cls_list) == len(boxes_list)
        assert len(tr_list) == 0 or len(tr_list) == len(boxes_list)
        if len(scores_list) == 0:
            scores_list = None
        if len(cls_list) == 0:
            cls_list = None
        if len(tr_list) == 0:
            tr_list = None
        return boxes_list, scores_list, cls_list, tr_list


def filter_boxes(
    boxes: torch.Tensor, min_area: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter a set of 2D bounding boxes given a minimum area.
    Args:

    Returns:
        Tuple[Tensor, Tensor]: filtered boxes, boolean mask
    """
    if min_area > 0.0:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        valid_mask = w * h >= min_area
        if not valid_mask.all():
            return boxes[valid_mask], valid_mask
    return boxes, boxes.new_ones((len(boxes),), dtype=torch.bool)


class Boxes3D(Boxes):
    """Container class for 3D boxes.

    boxes: torch.FloatTensor: (N, [9, 10]) where each entry is defined by
    [x, y, z, h, w, l, rx, ry, rz, Optional[score]].
    class_ids: torch.LongTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.LongTensor (N,) where each entry is the track id of
    the respective box.

    x,y,z are in OpenCV camera coordinate system. h, w, l are the 3D box
    dimensions and correspond to their respective axis (length first (x),
    height second (y), width last (z). The rotations are axis angles w.r.t.
    each axis (x,y,z).
    """

    @property
    def score(self) -> Optional[torch.Tensor]:
        """Return scores of 3D bounding boxes as tensor."""
        if not self.boxes.shape[-1] == 10:
            return None
        return self.boxes[:, -1]

    @property
    def center(self) -> torch.Tensor:
        """Return center of 3D bounding boxes as tensor."""
        return self.boxes[:, :3]

    @property
    def dimensions(self) -> torch.Tensor:
        """Return (h, w, l) of 3D bounding boxes as tensor."""
        return self.boxes[:, 3:6]

    @property
    def rot_x(self) -> Optional[torch.Tensor]:
        """Return rotation in x direction of 3D bounding boxes as tensor."""
        return self.boxes[:, 6]

    @property
    def rot_y(self) -> torch.Tensor:
        """Return rotation in y direction of 3D bounding boxes as tensor."""
        return self.boxes[:, 7]

    @property
    def rot_z(self) -> Optional[torch.Tensor]:
        """Return rotation in z direction of 3D bounding boxes as tensor."""
        return self.boxes[:, 8]

    @property
    def orientation(self) -> Optional[torch.Tensor]:
        """Return full orientation of 3D bounding boxes as tensor."""
        return self.boxes[:, 6:9]

    @classmethod
    def empty(
        cls: Type["TBoxes"], device: Optional[torch.device] = None
    ) -> "TBoxes":
        """Return empty boxes on device."""
        return cls(torch.empty(0, 10), torch.empty(0), torch.empty(0)).to(
            device
        )

    @classmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> "Boxes3D":
        """Convert from scalabel format to internal."""
        box_list, cls_list, idx_list = [], [], []
        has_class_ids = all((b.category is not None for b in labels))
        for i, label in enumerate(labels):
            box, score, box_cls, l_id = (
                label.box3d,
                label.score,
                label.category,
                label.id,
            )
            if box is None:
                continue
            if has_class_ids:
                if box_cls in class_to_idx:
                    cls_list.append(class_to_idx[box_cls])
                else:  # pragma: no cover
                    continue

            if score is None:
                box_list.append(
                    [*box.location, *box.dimension, *box.orientation]
                )
            else:
                box_list.append(
                    [*box.location, *box.dimension, *box.orientation, score]
                )
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)

        if len(box_list) == 0:  # pragma: no cover
            return cls.empty()
        box_tensor = torch.tensor(box_list, dtype=torch.float32)
        class_ids = (
            torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
        )
        track_ids = torch.tensor(idx_list, dtype=torch.long)
        return Boxes3D(box_tensor, class_ids, track_ids)

    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> List[Label]:
        """Convert from internal to scalabel format."""
        labels = []
        for i in range(len(self.boxes)):
            if self.track_ids is not None:
                label_id = str(self.track_ids[i].item())
            else:
                label_id = str(i)

            rx = float(self.boxes[i, 6])
            ry = float(self.boxes[i, 7])
            rz = float(self.boxes[i, 8])
            if self.boxes.shape[-1] == 10:
                score: Optional[float] = float(self.boxes[i, 9])
            else:
                score = None

            box = Box3D(
                location=[
                    float(self.boxes[i, 0]),
                    float(self.boxes[i, 1]),
                    float(self.boxes[i, 2]),
                ],
                dimension=[
                    float(self.boxes[i, 3]),
                    float(self.boxes[i, 4]),
                    float(self.boxes[i, 5]),
                ],
                orientation=[rx, ry, rz],
                alpha=-1.0,
            )
            label_dict = dict(id=label_id, box3d=box, score=score)

            if idx_to_class is not None:
                cls = idx_to_class[int(self.class_ids[i])]
            else:
                cls = str(int(self.class_ids[i]))  # pragma: no cover
            label_dict["category"] = cls
            labels.append(Label(**label_dict))

        return labels

    def transform(
        self, extrinsics: Extrinsics, in_image_frame: bool = False
    ) -> None:
        """Transform Boxes3D with given Extrinsics.

        Note: Mutates current Boxes3D.
        """
        if len(extrinsics) > 1:
            raise ValueError(
                f"Expected single Extrinsics but got len {len(extrinsics)}!"
            )

        center = torch.cat(
            [self.center, torch.ones_like(self.boxes[:, 0:1])], -1
        ).unsqueeze(-1)
        self.boxes[:, :3] = (extrinsics.tensor @ center)[:, :3, 0]
        rot = extrinsics.rotation @ euler_angles_to_matrix(self.orientation)

        if in_image_frame:
            # we use XZY convention here, since Z usually points up, but we
            # assume OpenCV cam coordinates (Y points down).
            self.boxes[:, 6:9] = matrix_to_euler_angles(rot, "XZY")[
                :, [0, 2, 1]
            ]
        else:
            self.boxes[:, 6:9] = matrix_to_euler_angles(rot)
