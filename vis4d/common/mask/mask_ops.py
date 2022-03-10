"""Mask ops."""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# implementation modified from:
# https://github.com/facebookresearch/detectron2/
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
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (
            slice(y0_int, y1_int),
            slice(x0_int, x1_int),
        )
    return img_masks[:, 0], ()  # pragma: no cover


def paste_masks_in_image(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    image_shape: Tuple[int, int],
    threshold: float = 0.5,
    bytes_per_float: int = 4,
    gpu_mem_limit: int = 1024 ** 3,
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
