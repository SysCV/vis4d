"""Utility functions for segmentation masks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _do_paste_mask(  # type: ignore
    masks: Tensor,
    boxes: Tensor,
    img_h: int,
    img_w: int,
    skip_empty: bool = True,
) -> tuple[Tensor, tuple[slice, slice] | tuple[()]]:
    """Paste mask onto image.

    On GPU, paste all masks together (up to chunk size) by using the entire
    image to sample the masks Compared to pasting them one by one, this has
    more operations but is faster on COCO-scale dataset.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): Masks with shape [N, 1, Hmask, Wmask].
        boxes (Tensor): Boxes with shape [N, 4].
        img_h (int): Image height.
        img_w (int): Image width.
        skip_empty (bool, optional): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU. Defaults to True.

    Returns:
        Tensor: Mask with shape [N, Himg, Wimg] if skip_empty == True, or
            a mask of shape (N, H', W') and the slice object for the
            corresponding region if skip_empty == False.
    """
    device = masks.device

    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1, min=0
        ).to(dtype=torch.int32)
        x0_int, y0_int = x0_int.item(), y0_int.item()
        x1_int = (
            torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w)
            .to(dtype=torch.int32)
            .item()
        )
        y1_int = (
            torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h)
            .to(dtype=torch.int32)
            .item()
        )
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    num_masks = masks.shape[0]

    img_y: Tensor = (
        torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    )
    img_x: Tensor = (
        torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    )
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1  # (N, h)
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1  # (N, w)

    gx = img_x[:, None, :].expand(num_masks, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(num_masks, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not masks.dtype.is_floating_point:
        masks = masks.float()
    img_masks = F.grid_sample(masks, grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (  # pylint: disable=unsubscriptable-object
            slice(y0_int, y1_int),
            slice(x0_int, x1_int),
        )
    return img_masks[:, 0], ()  # pylint: disable=unsubscriptable-object


def paste_masks_in_image(
    masks: Tensor,
    boxes: Tensor,
    image_shape: tuple[int, int],
    threshold: float = 0.5,
    bytes_per_float: int = 4,
    gpu_mem_limit: int = 1024**3,
) -> Tensor:
    """Paste masks that are of a fixed resolution into an image.

    The location, height, and width for pasting each mask is determined by
    their corresponding bounding boxes in boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): Masks with shape [N, Hmask, Wmask], where N is
            the number of detected object instances in the image and Hmask,
            Wmask are the mask width and mask height of the predicted mask
            (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Tensor): Boxes with shape [N, 4]. boxes[i] and masks[i]
            correspond to the same object instance.
        image_shape (tuple[int, int]): Image resolution (width, height).
        threshold (float, optional): Threshold for discretization of mask.
            Defaults to 0.5.
        bytes_per_float (int, optional): Number of bytes per float. Defaults to
            4.
        gpu_mem_limit (int, optional): GPU memory limit. Defaults to 1024**3.

    Returns:
        Tensor: Masks with shape [N, Himage, Wimage], where N is the
            number of detected object instances and Himage, Wimage are the
            image width and height.
    """
    assert (
        masks.shape[-1] == masks.shape[-2]
    ), "Only square mask predictions are supported"
    assert threshold >= 0
    num_masks = len(masks)
    if num_masks == 0:
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
        (
            masks_chunk,
            spatial_inds,
        ) = _do_paste_mask(
            masks[inds, None, :, :],
            boxes[inds, :4],
            img_h,
            img_w,
            skip_empty=masks.device.type == "cpu",
        )
        masks_chunk = torch.greater_equal(masks_chunk, threshold).to(
            dtype=torch.bool
        )
        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks.type(torch.uint8)


def nhw_to_hwc_mask(
    masks: Tensor, class_ids: Tensor, ignore_class: int = 255
) -> Tensor:
    """Convert N binary HxW masks to HxW semantic mask.

    Args:
        masks (Tensor): Masks with shape [N, H, W].
        class_ids (Tensor): Class IDs with shape [N, 1].
        ignore_class (int, optional): Ignore label. Defaults to 255.

    Returns:
        Tensor: Masks with shape [H, W], where each location indicate the
            class label.
    """
    hwc_mask = torch.full(
        masks.shape[1:], ignore_class, dtype=masks.dtype, device=masks.device
    )
    for mask, cat_id in zip(masks, class_ids):
        hwc_mask[mask > 0] = cat_id
    return hwc_mask


def clip_mask(mask: Tensor, target_shape: tuple[int, int]) -> Tensor:
    """Clip mask.

    Args:
        mask (Tensor): Mask with shape [C, H, W].
        target_shape (tuple[int, int]): Target shape (Ht, Wt).

    Returns:
        Tensor: Clipped mask with shape [C, Ht, Wt].
    """
    return mask[:, : target_shape[0], : target_shape[1]]


def remove_overlap(mask: Tensor, score: Tensor) -> Tensor:
    """Remove overlapping pixels between masks.

    Args:
        mask (Tensor): Mask with shape [N, H, W].
        score (Tensor): Score with shape [N].

    Returns:
        Tensor: Mask with shape [N, H, W].
    """
    foreground = torch.zeros(
        mask.shape[1:], dtype=torch.bool, device=mask.device
    )
    sort_idx = score.argsort(descending=True)
    for i in sort_idx:
        mask[i] = torch.logical_and(mask[i], ~foreground)
        foreground = torch.logical_or(mask[i], foreground)
    return mask


def postprocess_segms(
    segms: Tensor,
    images_hw: list[tuple[int, int]],
    original_hw: list[tuple[int, int]],
) -> Tensor:
    """Postprocess segmentations.

    Args:
        segms (Tensor): Segmentations with shape [B, C, H, W].
        images_hw (list[tuple[int, int]]): Image resolutions.
        original_hw (list[tuple[int, int]]): Original image resolutions.

    Returns:
        Tensor: Post-processed segmentations.
    """
    post_segms = []
    for segm, image_hw, orig_hw in zip(segms, images_hw, original_hw):
        post_segms.append(
            F.interpolate(
                segm[:, : image_hw[0], : image_hw[1]].unsqueeze(1),
                size=(orig_hw[0], orig_hw[1]),
                mode="bilinear",
            ).squeeze(1)
        )
    return torch.stack(post_segms).argmax(dim=1)


def masks2boxes(masks: Tensor) -> Tensor:
    """Obtain the tight bounding boxes of binary masks.

    Args:
        masks (Tensor): Binary mask of shape (N, H, W).

    Returns:
        Tensor: Boxes with shape (N, 4) of positive region in binary mask.
    """
    num_masks = masks.shape[0]
    bboxes = masks.new_zeros((num_masks, 4), dtype=torch.float32)
    x_any = torch.any(masks, dim=1)
    y_any = torch.any(masks, dim=2)
    for i in range(num_masks):
        x = torch.where(x_any[i, :])[0]
        y = torch.where(y_any[i, :])[0]
        if len(x) > 0 and len(y) > 0:
            bboxes[i, :] = bboxes.new_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1]
            )
    return bboxes
