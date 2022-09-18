"""Utility functions for segmentation masks."""
import torch


def nhw_to_hwc_mask(
    masks: torch.Tensor, class_ids: torch.Tensor, ignore_class: int = 255
) -> torch.Tensor:
    """Convert N binary HxW masks to HxW semantic mask."""
    hwc_mask = torch.full(
        masks.shape[1:], ignore_class, dtype=masks.dtype, device=masks.device
    )
    for mask, cat_id in zip(masks, class_ids):
        hwc_mask[mask > 0] = cat_id
    return hwc_mask
