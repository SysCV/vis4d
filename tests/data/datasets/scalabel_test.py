"""Tests for scalabel dataset."""
import os

import torch

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.scalabel import Scalabel


def test_3d_data() -> None:
    """Test 3D bounding box data from scalabel."""
    data_root = get_test_data("kitti_test")
    annotations = os.path.join(data_root, "labels/tracking_training.json")
    dataset = Scalabel(
        data_root,
        annotations,
        targets_to_load=(
            K.boxes3d,
            K.boxes3d_classes,
            K.boxes3d_track_ids,
        ),
    )
    assert len(dataset) == 4
    item = dataset[0]
    assert len(item[K.boxes3d]) == 7
    assert torch.isclose(
        item[K.boxes3d_classes],
        torch.tensor([2, 2, 2, 2, 2, 2, 2], dtype=torch.long),
    ).all()
    assert item[K.original_hw] == (375, 1242)

    assert torch.isclose(
        item[K.boxes3d_track_ids],
        torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long),
    ).all()
