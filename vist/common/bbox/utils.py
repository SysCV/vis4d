"""Utility functions for bounding boxes."""
import torch

from functools import partial
from six.moves import map, zip

from detectron2.structures import Boxes, pairwise_iou

from vist.struct import Boxes2D


def compute_iou(boxes1: Boxes2D, boxes2: Boxes2D) -> torch.Tensor:
    """Compute IoU between all pairs of boxes.

    Args:
        boxes1, boxes2 (Boxes2D): Contains N & M boxes.

    Returns:
        Tensor: IoU, size [N, M].
    """
    return pairwise_iou(Boxes(boxes1.boxes[:, :4]), Boxes(boxes2.boxes[:, :4]))


def random_choice(tensor: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Randomly choose elements from a tensor."""
    perm = torch.randperm(len(tensor), device=tensor.device)[:sample_size]
    return tensor[perm]


def non_intersection(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Get the elements of t1 that are not present in t2."""
    compareview = t2.repeat(t1.shape[0], 1).T
    return t1[(compareview != t1).T.prod(1) == 1]


def multi_apply(func, *args, **kwargs):
    """Multi-apply one same function."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
