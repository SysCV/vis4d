"""OpenMT Label data structures."""
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops.roi_align import roi_align
from pycocotools import mask as mask_utils
from scalabel.label.transforms import mask_to_box2d, poly2ds_to_mask
from scalabel.label.typing import Box2D, Box3D, ImageSize, Label

from ..common.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)
from .data import Extrinsics
from .structures import DataInstance, LabelInstance, NDArrayUI8
from .utils import do_paste_mask

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

    def transform(self, extrinsics: Extrinsics) -> None:
        """Transform Boxes3D with given Extrinsics.

        Note: Mutates current Boxes3D.
        """
        if len(extrinsics) > 1:
            raise ValueError(
                f"Expected single Extrinsics but got len {len(extrinsics)}!"
            )

        center = torch.cat(
            [self.center, torch.ones_like(self.boxes[:, 0:1])], -1
        )
        self.boxes[:, :3] = (center @ extrinsics.transpose().tensor[0])[:, :3]
        rot = extrinsics.rotation @ euler_angles_to_matrix(self.orientation)
        # we use XZY convention here, since Z usually points up, but we assume
        # OpenCV cam coordinates (Y points down).
        self.boxes[:, 6:9] = matrix_to_euler_angles(rot, "XZY")[:, [0, 2, 1]]


class Masks(LabelInstance):
    """Container class for segmentation masks.

    masks: torch.ByteTensor: (N, H, W) where each entry is a binary mask
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
        scores: torch.Tensor = None,
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
        if scores is not None:
            assert isinstance(scores, torch.Tensor)
            assert len(masks) == len(scores)
            assert masks.device == scores.device

        self.masks = masks
        self.class_ids = class_ids
        self.track_ids = track_ids
        self.scores = scores
        self.metadata = metadata

    @property
    def height(self) -> int:
        """Return height of masks."""
        return self.masks.size(1)  # type: ignore

    @property
    def width(self) -> int:
        """Return width of masks."""
        return self.masks.size(2)  # type: ignore

    @property
    def size(self) -> Tuple[int, int]:
        """Return size of masks (w, h)."""
        return self.width, self.height

    def resize(self, out_size: Tuple[int, int]) -> None:
        """Resize masks according to factor."""
        width, height = out_size
        self.masks = F.interpolate(
            self.masks.unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)

    def crop_and_resize(
        self,
        bboxes: torch.Tensor,
        out_shape: Tuple[int, int],
        inds: torch.Tensor,
        device: Optional[str] = "cpu",
        binarize: Optional[bool] = True,
    ) -> "Masks":
        """Crop and resize masks with input bboxes."""
        if len(self) == 0:  # pragma: no cover
            return self

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(num_bbox, device=device).to(
            dtype=bboxes.dtype
        )[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = self.masks.index_select(0, inds).to(dtype=rois.dtype)
            targets = roi_align(
                gt_masks_th[:, None, :, :],
                rois,
                out_shape,
                1.0,
                0,
                "avg",
                True,
            ).squeeze(1)
            if binarize:
                resized_masks = targets >= 0.5
            else:
                resized_masks = targets
        else:
            resized_masks = []  # pragma: no cover
        return type(self)(resized_masks)

    def paste_masks_in_image(
        self,
        boxes: Boxes2D,
        image_shape: Tuple[int, int],
        threshold: float = 0.5,
        bytes_per_float: int = 4,
        gpu_mem_limit: int = 1024 ** 3,
    ) -> None:
        """Paste masks that are of a fixed resolution into an image.

        This implementation is modified from
        https://github.com/facebookresearch/detectron2/
        """
        assert (
            self.masks.shape[-1] == self.masks.shape[-2]
        ), "Only square mask predictions are supported"
        num_masks = len(self.masks)
        if num_masks == 0:  # pragma: no cover
            return
        assert len(boxes) == num_masks, boxes.boxes.shape

        img_w, img_h = image_shape

        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if self.device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True so that it performs minimal number of operations.
            num_chunks = num_masks
        else:  # pragma: no cover
            # GPU benefits from parallelism for larger chunks, but may have
            # memory issue int(img_h) because shape may be tensors in tracing
            num_chunks = int(
                np.ceil(
                    num_masks
                    * int(img_h)
                    * int(img_w)
                    * bytes_per_float
                    / gpu_mem_limit
                )
            )
            assert (
                num_chunks <= num_masks
            ), "Default gpu_mem_limit is too small; try increasing it"
        chunks = torch.chunk(
            torch.arange(num_masks, device=self.device), num_chunks
        )

        img_masks = torch.zeros(
            num_masks,
            img_h,
            img_w,
            device=self.device,
            dtype=torch.bool if threshold >= 0 else torch.uint8,
        )
        for inds in chunks:
            (masks_chunk, spatial_inds,) = do_paste_mask(
                self.masks[inds, None, :, :],
                boxes.boxes[inds, :4],
                img_h,
                img_w,
                skip_empty=self.device.type == "cpu",
            )

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(  # pragma: no cover
                    dtype=torch.uint8
                )

            img_masks[(inds,) + spatial_inds] = masks_chunk
        self.masks = img_masks.type(torch.uint8)

    def __getitem__(self: "Masks", item) -> "Masks":  # type: ignore
        """Shadows tensor based indexing while returning new Masks."""
        if isinstance(item, tuple):  # pragma: no cover
            item = item[0]
        masks = self.masks[item]
        class_ids = (
            self.class_ids[item] if self.class_ids is not None else None
        )
        track_ids = (
            self.track_ids[item] if self.track_ids is not None else None
        )
        scores = self.scores[item] if self.scores is not None else None
        if len(masks.shape) < 3:
            if class_ids is not None:
                class_ids = class_ids.view(1, -1)
            if track_ids is not None:
                track_ids = track_ids.view(1, -1)
            if scores is not None:
                scores = scores.view(1, -1)
            return type(self)(
                masks.view(1, masks.size(0), masks.size(1)),
                class_ids,
                track_ids,
                scores,
                self.metadata,
            )

        return type(self)(masks, class_ids, track_ids, scores, self.metadata)

    @classmethod
    def from_scalabel(
        cls,
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
    ) -> Tuple["Masks", "Boxes2D"]:
        """Convert from scalabel format to internal."""
        box_list, bitmask_list, cls_list, idx_list = [], [], [], []
        score_list = []
        has_class_ids = all((b.category is not None for b in labels))
        has_scores = all((b.score is not None for b in labels))
        for i, label in enumerate(labels):
            if label.poly2d is None and label.rle is None:
                continue
            if label.rle is not None:
                bitmask = mask_utils.decode(dict(label.rle))
            elif label.poly2d is not None:  # pragma: no cover
                assert (
                    image_size is not None
                ), "image size must be specified for masks with polygons!"
                bitmask_raw = poly2ds_to_mask(image_size, label.poly2d)
                bitmask: NDArrayUI8 = (bitmask_raw > 0).astype(  # type: ignore
                    bitmask_raw.dtype
                )
            if np.count_nonzero(bitmask) == 0:  # pragma: no cover
                continue
            bitmask_list.append(bitmask)
            box = mask_to_box2d(bitmask)
            mask_cls, l_id, score = label.category, label.id, label.score
            if score is None:
                box_list.append([box.x1, box.y1, box.x2, box.y2])
            else:
                box_list.append([box.x1, box.y1, box.x2, box.y2, score])

            if has_class_ids:
                cls_list.append(class_to_idx[mask_cls])  # type: ignore
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)
            if has_scores:
                score_list.append(score)

        box_tensor = torch.tensor(box_list, dtype=torch.float32)
        mask_tensor = torch.tensor(bitmask_list, dtype=torch.uint8)
        class_ids = (
            torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
        )
        track_ids = torch.tensor(idx_list, dtype=torch.long)
        scores = (
            torch.tensor(score_list, dtype=torch.float32)
            if has_scores
            else None
        )
        return Masks(mask_tensor, class_ids, track_ids, scores), Boxes2D(
            box_tensor, class_ids, track_ids
        )

    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> List[Label]:
        """Convert from internal to scalabel format."""
        labels = []
        for i, mask in enumerate(self.masks):
            if idx_to_class is not None:
                cls = idx_to_class[int(self.class_ids[i])]
            else:
                cls = str(int(self.class_ids[i]))  # pragma: no cover
            if self.track_ids is not None:
                label_id = str(self.track_ids[i].item())
            else:
                label_id = str(i)
            score = None
            if self.scores is not None:
                score = self.scores[i].item()
            rle = mask_utils.encode(
                np.array(
                    mask[:, :, None].cpu().numpy(),
                    order="F",
                    dtype="uint8",
                )
            )[0]
            rle_label = dict(
                counts=rle["counts"].decode("utf-8"), size=rle["size"]
            )
            label_dict = dict(
                id=label_id, category=cls, score=score, rle=rle_label
            )
            labels.append(Label(**label_dict))

        return labels

    def to_ndarray(self) -> NDArrayUI8:
        """Convert masks to ndarray."""
        return self.masks.cpu().numpy()  # type: ignore

    def __len__(self) -> int:
        """Get length of the object."""
        return len(self.masks)

    def clone(self: "Masks") -> "Masks":
        """Create a copy of the object."""
        class_ids = (
            self.class_ids.clone() if self.class_ids is not None else None
        )
        track_ids = (
            self.track_ids.clone() if self.track_ids is not None else None
        )
        scores = self.scores.clone() if self.scores is not None else None
        return type(self)(
            self.masks.clone(), class_ids, track_ids, scores, self.metadata
        )

    def to(self: "Masks", device: torch.device) -> "Masks":
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
        scores = (
            self.scores.to(device=device) if self.scores is not None else None
        )
        return type(self)(
            self.masks.to(device=device),
            class_ids,
            track_ids,
            scores,
            self.metadata,
        )

    @property
    def device(self) -> torch.device:
        """Get current device of data."""
        return self.masks.device
