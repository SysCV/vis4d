"""Box2D test file."""

import torch

from vis4d.op.box.box2d import (
    bbox_intersection,
    bbox_iou,
    filter_boxes_by_area,
)


def test_bbox_intersection() -> None:
    """Test bbox_intersection function."""
    box1 = torch.tensor([[0, 0, 10, 10], [2, 2, 20, 20]])
    box2 = torch.tensor([[5, 5, 10, 10], [3, 1, 6, 4]])
    inter = bbox_intersection(box1, box2)
    assert inter[0, 0] == 25
    assert inter[0, 1] == 9
    assert inter[1, 0] == 25
    assert inter[1, 1] == 6


def test_bbox_iou() -> None:
    """Test bbox_iou function."""
    box1 = torch.tensor([[0, 0, 10, 10], [2, 2, 20, 20]])
    box2 = torch.tensor([[5, 5, 10, 10], [3, 1, 6, 4]])
    ious = bbox_iou(box1, box2)
    assert ious[0, 0] == 25 / 100
    assert ious[0, 1] == 9 / 100
    assert ious[1, 0] == 25 / 324
    assert ious[1, 1] == 6 / 327


def test_filter_boxes_by_area() -> None:
    """Test filter_boxes_by_area function."""
    boxes = torch.tensor(
        [[0, 0, 10, 10], [2, 2, 20, 20], [5, 5, 10, 10], [3, 1, 6, 4]]
    )
    new_boxes, mask = filter_boxes_by_area(boxes, 20)
    assert new_boxes.shape[0] == 3
    assert (new_boxes == boxes[:3]).all()
    assert (new_boxes == boxes[mask]).all()
    assert mask.sum() == 3
