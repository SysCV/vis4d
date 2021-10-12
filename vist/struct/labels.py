"""OpenMT Label data structures."""
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from scalabel.label.transforms import mask_to_box2d, poly2ds_to_mask
from scalabel.label.typing import Box2D, Box3D, Label, ImageSize

from .structures import DataInstance, LabelInstance

TBoxes = TypeVar("TBoxes", bound="Boxes")


class Boxes(DataInstance):
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
        metadata: Optional[Dict[str, Union[bool, int, float, str]]] = None,
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
        self.metadata = metadata

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
            return type(self)(
                boxes.view(1, -1), class_ids, track_ids, self.metadata
            )

        return type(self)(boxes, class_ids, track_ids, self.metadata)

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
        return type(self)(
            self.boxes.clone(), class_ids, track_ids, self.metadata
        )

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
        return type(self)(
            self.boxes.to(device=device), class_ids, track_ids, self.metadata
        )

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


class Boxes2D(Boxes, LabelInstance):
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
        area = (self.boxes[:, 2] - self.boxes[:, 0]) * (
            self.boxes[:, 3] - self.boxes[:, 1]
        )
        return area

    @classmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> Tuple["Boxes2D"]:
        """Convert from scalabel format to internal."""
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

            if score is None:
                box_list.append([box.x1, box.y1, box.x2, box.y2])
            else:
                box_list.append([box.x1, box.y1, box.x2, box.y2, score])

            if has_class_ids:
                cls_list.append(class_to_idx[box_cls])  # type: ignore
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)

        box_tensor = torch.tensor(box_list, dtype=torch.float32)
        class_ids = (
            torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
        )
        track_ids = torch.tensor(idx_list, dtype=torch.long)
        return (Boxes2D(box_tensor, class_ids, track_ids),)

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
            box = Box2D(
                x1=float(self.boxes[i, 0]),
                y1=float(self.boxes[i, 1]),
                x2=float(self.boxes[i, 2]),
                y2=float(self.boxes[i, 3]),
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


class Boxes3D(Boxes, LabelInstance):
    """Container class for 3D boxes.

    boxes: torch.FloatTensor: (N, [7, 8]) where each entry is defined as
    [x, y, z, h, w, l, ry, Optional[score]] or (N, [9, 10]) where each entry
    is defined by [x, y, z, h, w, l, rx, ry, rz, Optional[score]].
    class_ids: torch.LongTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.LongTensor (N,) where each entry is the track id of
    the respective box.

    x,y,z are in OpenCV camera coordinate system. l, h, w, are the 3D box
    dimensions and correspond to their respective axis (length first (x),
    height second (y), width last (z). The rotations are axis angles w.r.t.
    each axis (x,y,z).
    """

    @property
    def score(self) -> Optional[torch.Tensor]:
        """Return scores of 3D bounding boxes as tensor."""
        if not self.boxes.shape[-1] in [8, 10]:
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
        if self.boxes.shape[-1] in [7, 8]:
            return None
        return self.boxes[:, 6]

    @property
    def rot_y(self) -> torch.Tensor:
        """Return rotation in y direction of 3D bounding boxes as tensor."""
        if self.boxes.shape[-1] in [7, 8]:
            return self.boxes[:, 6]
        return self.boxes[:, 7]

    @property
    def rot_z(self) -> Optional[torch.Tensor]:
        """Return rotation in z direction of 3D bounding boxes as tensor."""
        if self.boxes.shape[-1] in [7, 8]:
            return None
        return self.boxes[:, 8]

    @classmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> Tuple["Boxes3D"]:
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

            if score is None:
                box_list.append(
                    [*box.location, *box.dimension, *box.orientation]
                )
            else:
                box_list.append(
                    [*box.location, *box.dimension, *box.orientation, score]
                )
            if has_class_ids:
                cls_list.append(class_to_idx[box_cls])  # type: ignore
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)

        box_tensor = torch.tensor(box_list, dtype=torch.float32)
        class_ids = (
            torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
        )
        track_ids = torch.tensor(idx_list, dtype=torch.long)
        return (Boxes3D(box_tensor, class_ids, track_ids),)

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

            if self.boxes.shape[-1] < 9:
                rx = 0.0
                ry = float(self.boxes[i, 6])
                rz = 0.0
                if self.boxes.shape[-1] == 8:
                    score: Optional[float] = float(self.boxes[i, 7])
                else:
                    score = None
            else:
                rx = float(self.boxes[i, 6])
                ry = float(self.boxes[i, 7])
                rz = float(self.boxes[i, 8])
                if self.boxes.shape[-1] == 10:
                    score = float(self.boxes[i, 9])
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


class Bitmasks(LabelInstance):  # type: ignore
    """Container class for bitmasks.

    masks: torch.FloatTensor: (N, W, H) where each entry is a binary mask
    class_ids: torch.LongTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.LongTensor (N,) where each entry is the track id of
    the respective box.
    """

    def __init__(
        self,
        masks: torch.Tensor,
        class_ids: torch.Tensor = None,
        track_ids: torch.Tensor = None,
        metadata: Optional[Dict[str, Union[bool, int, float, str]]] = None,
    ) -> None:
        """Init."""
        assert isinstance(masks, torch.Tensor) and len(masks.shape) == 3
        if class_ids is not None:
            assert isinstance(class_ids, torch.Tensor)
            assert len(masks) == len(class_ids)
            assert masks.device == class_ids.device
        if track_ids is not None:
            assert isinstance(track_ids, torch.Tensor)
            assert len(masks) == len(track_ids)
            assert masks.device == track_ids.device

        self.masks = masks
        self.class_ids = class_ids
        self.track_ids = track_ids
        self.metadata = metadata

    @classmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> Tuple["Bitmasks", "Boxes2D"]:
        """Convert from scalabel format to internal."""
        assert (
            image_size is not None
        ), "image size must be specified for bitmasks!"
        box_list, bitmask_list, cls_list, idx_list = [], [], [], []
        has_class_ids = all((b.category is not None for b in labels))
        for i, label in enumerate(labels):
            if label.poly2d is None:
                continue
            poly2d = label.poly2d
            bitmask_raw = poly2ds_to_mask(image_size, poly2d)
            bitmask = (bitmask_raw > 0).astype(bitmask_raw.dtype)
            bitmask_list.append(bitmask)
            bbox = mask_to_box2d(bitmask)  # type: ignore
            box_list.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            mask_cls, l_id = label.category, label.id

            if has_class_ids:
                cls_list.append(class_to_idx[mask_cls])  # type: ignore
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)

        box_tensor = torch.tensor(box_list, dtype=torch.float32)
        mask_tensor = torch.tensor(bitmask_list, dtype=torch.uint8)
        class_ids = (
            torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
        )
        track_ids = torch.tensor(idx_list, dtype=torch.long)
        return Bitmasks(mask_tensor, class_ids, track_ids), Boxes2D(
            box_tensor, class_ids, track_ids
        )

    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> List[Label]:
        """Convert from internal to scalabel format."""
        raise NotImplementedError

    def to(self: "Bitmasks", device: torch.device) -> "Bitmasks":
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
        return type(self)(
            self.masks.to(device=device),
            class_ids,
            track_ids,
            self.metadata,
        )

    @property
    def device(self) -> torch.device:
        """Get current device of data."""
        return self.masks.device
