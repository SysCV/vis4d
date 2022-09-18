"""Vis4D Masks data structures."""
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from scalabel.label.transforms import mask_to_rle, poly2ds_to_mask, rle_to_mask
from scalabel.label.typing import ImageSize, Label
from torchvision.ops import roi_align

# from vis4d.op.detect.mask import paste_masks_in_image

from ..structures import LabelInstance, NDArrayUI8
from .boxes import Boxes2D

TMasks = TypeVar("TMasks", bound="Masks")


# implementation modified from:
# https://github.com/facebookresearch/detectron2/
# TODO (Thomas) describe the changes
# TODO: copying here to avoid circular import
def _do_paste_mask(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    img_h: int,
    img_w: int,
    skip_empty: bool = True,
) -> torch.Tensor:
    """Paste mask onto image."""
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1, min=0
        ).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(
            dtype=torch.int32
        )
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(
            dtype=torch.int32
        )
    else:  # pragma: no cover
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    num_masks = masks.shape[0]

    img_y = (
        torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    )
    img_x = (
        torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    )
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(num_masks, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(num_masks, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not masks.dtype.is_floating_point:
        masks = masks.float()
    img_masks = F.grid_sample(masks, grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    return img_masks[:, 0], ()  # pragma: no cover


def paste_masks_in_image(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    image_shape: Tuple[int, int],
    threshold: float = 0.5,
    bytes_per_float: int = 4,
    gpu_mem_limit: int = 1024**3,
) -> torch.Tensor:
    """Paste masks that are of a fixed resolution into an image.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/
    """
    assert (
        masks.shape[-1] == masks.shape[-2]
    ), "Only square mask predictions are supported"
    assert threshold >= 0
    num_masks = len(masks)
    if num_masks == 0:  # pragma: no cover
        return masks

    img_w, img_h = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if masks.device.type == "cpu":
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
        torch.arange(num_masks, device=masks.device), num_chunks
    )

    img_masks = torch.zeros(
        num_masks, img_h, img_w, device=masks.device, dtype=torch.bool
    )
    for inds in chunks:
        (masks_chunk, spatial_inds,) = _do_paste_mask(
            masks[inds, None, :, :],
            boxes[inds, :4],
            img_h,
            img_w,
            skip_empty=masks.device.type == "cpu",
        )
        masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks.type(torch.uint8)


class Masks(LabelInstance):
    """Abstract container for Instance / Semantic Masks.

    masks: torch.ByteTensor (N, H, W) where each entry is a binary mask
    class_ids: torch.LongTensor (N,) where each entry is the class id of mask.
    track_ids: torch.LongTensor (N,) where each entry is the track id of mask.
    score: torch.FloatTensor (N,) where each entry is the confidence score
    of mask.
    """

    def __init__(
        self,
        masks: torch.Tensor,
        class_ids: torch.Tensor = None,
        track_ids: torch.Tensor = None,
        score: torch.Tensor = None,
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
        if score is not None:
            assert isinstance(score, torch.Tensor)
            assert len(masks) == len(score)
            assert masks.device == score.device

        self.masks = masks
        self.class_ids = class_ids
        self.track_ids = track_ids
        self.score = score

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

    def resize(self, out_size: Tuple[int, int], mode: str = "nearest") -> None:
        """Resize masks according to out_size."""
        width, height = out_size
        self.masks = F.interpolate(
            self.masks.unsqueeze(1), size=(height, width), mode=mode
        ).squeeze(1)

    def unique(self) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        """Return unique elements in masks."""
        return self.masks.unique(return_counts=True)  # type: ignore

    @classmethod
    def empty(
        cls: Type["TMasks"], device: Optional[torch.device] = None
    ) -> "TMasks":
        """Return empty masks on device."""
        return cls(
            torch.empty(0, 720, 1280),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        ).to(device)

    @classmethod
    def from_scalabel(
        cls: Type["TMasks"],
        labels: List[Label],
        class_to_idx: Dict[str, int],
        label_id_to_idx: Optional[Dict[str, int]] = None,
        image_size: Optional[ImageSize] = None,
        bg_as_class: bool = False,
    ) -> "TMasks":
        """Convert from scalabel format to internal."""
        bitmask_list, cls_list, idx_list = [], [], []
        score_list = []
        has_class_ids = all((b.category is not None for b in labels))
        has_score = all((b.score is not None for b in labels))
        if bg_as_class:
            foreground: Optional[NDArrayUI8] = None
        for i, label in enumerate(labels):
            if label.poly2d is None and label.rle is None:
                continue
            mask_cls, l_id, score = label.category, label.id, label.score
            if has_class_ids:
                if mask_cls in class_to_idx:
                    cls_list.append(class_to_idx[mask_cls])
                else:  # pragma: no cover
                    continue
            if label.rle is not None:
                bitmask = rle_to_mask(label.rle)
            elif label.poly2d is not None:
                assert (
                    image_size is not None
                ), "image size must be specified for masks with polygons!"
                bitmask_raw = poly2ds_to_mask(image_size, label.poly2d)
                bitmask: NDArrayUI8 = (bitmask_raw > 0).astype(  # type: ignore
                    bitmask_raw.dtype
                )
            bitmask_list.append(bitmask)
            idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
            idx_list.append(idx)
            if has_score:
                score_list.append(score)
            if bg_as_class:
                foreground = (
                    bitmask
                    if foreground is None
                    else np.logical_or(foreground, bitmask)
                )
        if bg_as_class:
            assert foreground is not None
            bitmask_list.append(np.logical_not(foreground))
            idx_list.append(len(labels))
            if has_class_ids:
                assert "background" in class_to_idx, (
                    '"bg_as_class" requires "background" class to be '
                    "in category_mapping"
                )
                cls_list.append(class_to_idx["background"])
            if has_score:  # pragma: no cover
                score_list.append(1.0)
        if len(bitmask_list) == 0:  # pragma: no cover
            return cls.empty()
        mask_tensor = torch.tensor(np.array(bitmask_list), dtype=torch.uint8)
        class_ids = (
            torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
        )
        track_ids = torch.tensor(idx_list, dtype=torch.long)
        score = (
            torch.tensor(score_list, dtype=torch.float32)
            if has_score
            else None
        )
        return cls(mask_tensor, class_ids, track_ids, score)

    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> List[Label]:
        """Convert from internal to scalabel format."""
        labels = []
        for i, mask in enumerate(self.masks):
            if mask.sum().item() == 0:
                continue
            if idx_to_class is not None:
                cls = idx_to_class[int(self.class_ids[i])]
            else:
                cls = str(int(self.class_ids[i]))  # pragma: no cover
            if self.track_ids is not None:
                label_id = str(self.track_ids[i].item())
            else:
                label_id = str(i)
            score = None
            if self.score is not None:
                score = self.score[i].item()
            rle = mask_to_rle(mask.cpu().numpy())
            label_dict = dict(id=label_id, category=cls, score=score, rle=rle)
            labels.append(Label(**label_dict))

        return labels

    def __getitem__(self: "TMasks", item) -> "TMasks":  # type: ignore
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
        score = self.score[item] if self.score is not None else None
        if len(masks.shape) < 3:
            if class_ids is not None:
                class_ids = class_ids.view(1, -1)
            if track_ids is not None:
                track_ids = track_ids.view(1, -1)
            if score is not None:
                score = score.view(1, -1)
            return type(self)(
                masks.view(1, masks.size(0), masks.size(1)),
                class_ids,
                track_ids,
                score,
            )

        return type(self)(masks, class_ids, track_ids, score)

    def to_ndarray(self) -> NDArrayUI8:
        """Convert masks to ndarray."""
        return self.masks.cpu().numpy()  # type: ignore

    def __len__(self) -> int:
        """Get length of the object."""
        return len(self.masks)

    def clone(self: "TMasks") -> "TMasks":
        """Create a copy of the object."""
        class_ids = (
            self.class_ids.clone() if self.class_ids is not None else None
        )
        track_ids = (
            self.track_ids.clone() if self.track_ids is not None else None
        )
        score = self.score.clone() if self.score is not None else None
        return type(self)(self.masks.clone(), class_ids, track_ids, score)

    def to(self: "TMasks", device: torch.device) -> "TMasks":
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
        score = (
            self.score.to(device=device) if self.score is not None else None
        )
        return type(self)(
            self.masks.to(device=device),
            class_ids,
            track_ids,
            score,
        )

    @property
    def device(self) -> torch.device:
        """Get current device of data."""
        return self.masks.device

    def get_boxes2d(self) -> Boxes2D:
        """Return corresponding Boxes2D for the masks inside self."""
        if len(self) == 0:
            return Boxes2D.empty()

        boxes_list = []
        for i, mask in enumerate(self.masks):
            foreground = mask.nonzero()
            if len(foreground) == 0:  # pragma: no cover
                x1, y1, x2, y2 = (
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                )
            else:
                y1, x1 = foreground.min(dim=0)[0].float()
                y2, x2 = foreground.max(dim=0)[0].float()
            entries = [x1, y1, x2, y2]
            if self.score is not None:
                entries.append(self.score[i])
            boxes_list.append(torch.stack(entries))
        return Boxes2D(torch.stack(boxes_list), self.class_ids, self.track_ids)


class InstanceMasks(Masks):
    """Container class for instance segmentation masks.

    masks: torch.ByteTensor (N, H, W) or (N, H_mask, W_mask) where each
    entry is a binary mask and H/W_mask is a unified mask size, e.g., 28x28.
    class_ids: torch.LongTensor (N,) where each entry is the class id of mask.
    track_ids: torch.LongTensor (N,) where each entry is the track id of mask.
    score: torch.FloatTensor (N,) where each entry is the confidence score
    of mask.
    detections: Optional[Boxes2D] if masks is (N, H_mask, W_mask), detections
    are used to paste them back in the original resolution.
    """

    def __init__(
        self,
        masks: torch.Tensor,
        class_ids: torch.Tensor = None,
        track_ids: torch.Tensor = None,
        score: torch.Tensor = None,
        detections: Optional[Boxes2D] = None,
    ) -> None:
        """Init."""
        super().__init__(masks, class_ids, track_ids, score)
        if detections is not None:
            assert isinstance(detections, Boxes2D)
            assert len(masks) == len(detections)
            assert masks.device == detections.device
        self.detections = detections

    def crop_and_resize(
        self,
        boxes: Boxes2D,
        out_shape: Tuple[int, int],
        binarize: Optional[bool] = True,
    ) -> "InstanceMasks":
        """Crop and resize masks with input bboxes."""
        if len(self) == 0:
            return self

        assert len(boxes) == len(
            self.masks
        ), "Number of boxes should be the same as masks"
        fake_inds = torch.arange(len(boxes), device=boxes.device)[:, None]
        bboxes = (
            boxes.boxes[:, :-1] if boxes.score is not None else boxes.boxes
        )
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        gt_masks_th = self.masks[:, None, :, :].type(rois.dtype)
        targets = roi_align(
            gt_masks_th, rois, out_shape, 1.0, 0, True
        ).squeeze(1)
        if binarize:
            resized_masks = targets >= 0.5
        else:
            resized_masks = targets
        return InstanceMasks(resized_masks, detections=boxes)

    def postprocess(
        self,
        original_wh: Tuple[int, int],
        output_wh: Tuple[int, int],
        clip: bool = True,
        resolve_overlap: bool = True,
    ) -> None:
        """Postprocess instance masks."""
        if len(self) == 0:  # pragma: no cover
            return
        if self.size != output_wh:
            assert (
                self.detections is not None
            ), "Pasting masks requires detections to be specified!"
            self.masks = paste_masks_in_image(
                self.masks, self.detections.boxes, original_wh
            )
        if resolve_overlap:
            # remove overlaps in instance masks
            foreground = torch.zeros(
                self.masks.shape[1:], dtype=torch.bool, device=self.device
            )
            sort_idx = self.score.argsort(descending=True)
            for i in sort_idx:
                self.masks[i] = torch.logical_and(self.masks[i], ~foreground)
                foreground = torch.logical_or(self.masks[i], foreground)
        if self.size != original_wh:
            self.resize(original_wh)


class SemanticMasks(Masks):
    """Container class for semantic segmentation masks.

    masks: torch.ByteTensor (N, H, W) where each entry is a binary mask
    class_ids: torch.LongTensor (N,) where each entry is the class id of mask.
    """

    ignore_class: int = 255

    def crop(self, crop_box: Boxes2D) -> "SemanticMasks":
        """Crop semantic mask."""
        assert len(crop_box) == 1
        x1, y1, x2, y2 = crop_box.boxes[0].int().tolist()
        return SemanticMasks(self.masks[:, y1:y2, x1:x2], self.class_ids)

    def to_nhw_mask(self) -> "SemanticMasks":
        """Convert HxW semantic mask to N binary HxW masks."""
        return SemanticMasks.from_hwc_tensor(self.masks)

    @classmethod
    def from_hwc_tensor(cls, masks: torch.Tensor) -> "SemanticMasks":
        """Convert HxW semantic mask tensor to N binary HxW semantic masks."""
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)
        assert masks.size(0) == 1
        nhw_masks, cls_list = [], []
        for cat_id in torch.unique(masks):
            if cat_id == cls.ignore_class:
                continue
            nhw_masks.append((masks == cat_id).type(torch.uint8))
            cls_list.append(cat_id)
        return SemanticMasks(
            torch.cat(nhw_masks).type(torch.uint8),
            torch.tensor(cls_list, dtype=torch.long, device=masks.device),
        )

    def to_hwc_mask(self) -> torch.Tensor:
        """Convert N binary HxW masks to HxW semantic mask."""
        hwc_mask = torch.full(
            self.masks.shape[1:], self.ignore_class, device=self.device
        )
        for mask, cat_id in zip(self.masks, self.class_ids):
            hwc_mask[mask > 0] = cat_id
        return hwc_mask

    @classmethod
    def pad(
        cls, masks: List["SemanticMasks"], image_size: Tuple[int, int]
    ) -> List["SemanticMasks"]:
        """Pad each semantic mask to image_size."""
        pad_masks = []
        for mask in masks:
            pad_w = max(image_size[0] - mask.width, 0)
            pad_h = max(image_size[1] - mask.height, 0)
            if pad_w == 0 and pad_h == 0:
                pad_mask = mask
            else:
                pad_size = (0, pad_w, 0, pad_h)
                pad_mask_ = F.pad(
                    mask.masks.unsqueeze(0), pad_size, value=cls.ignore_class
                )[0]
                pad_mask = SemanticMasks(
                    pad_mask_, mask.class_ids, mask.track_ids, mask.score
                )
            pad_masks.append(pad_mask)
        return pad_masks

    def postprocess(
        self,
        original_wh: Tuple[int, int],
        output_wh: Tuple[int, int],
        clip: bool = True,
        resolve_overlap: bool = True,
    ) -> None:
        """Postprocess semantic masks."""
        self.masks = self.masks[:, : output_wh[1], : output_wh[0]]
        if original_wh != output_wh:
            self.resize(original_wh)


class MaskLogits(Masks):
    """Container class for mask logits."""

    def resize(
        self, out_size: Tuple[int, int], mode: str = "bilinear"
    ) -> None:
        """Resize masks according to out_size."""
        width, height = out_size
        self.masks = F.interpolate(
            self.masks.unsqueeze(0),
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        ).squeeze(0)

    def paste_masks(
        self, boxes: Boxes2D, out_size: Tuple[int, int]
    ) -> "MaskLogits":
        """Paste masks into an image."""
        return MaskLogits(
            paste_masks_in_image(self.masks, boxes.boxes, out_size)
        )

    def postprocess(
        self,
        original_wh: Tuple[int, int],
        output_wh: Tuple[int, int],
        clip: bool = True,
        resolve_overlap: bool = True,
    ) -> None:
        """Postprocess instance masks."""
        self.masks = self.masks[:, : output_wh[1], : output_wh[0]]
        if original_wh != output_wh:
            self.resize(original_wh)
