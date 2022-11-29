"""Tests for scalabel dataset."""
import torch

from vis4d.data.const import CommonKeys
from vis4d.unittest.util import get_test_file

from .scalabel import Scalabel


def test_len_getitem():
    """Test len / getitem methods of scalabel."""
    data_root = get_test_file("detect/bdd100k-samples/images", rel_path="run")
    annotations = get_test_file(
        "detect/bdd100k-samples/labels", rel_path="run"
    )
    config = get_test_file(
        "detect/bdd100k-samples/config.toml", rel_path="run"
    )
    dataset = Scalabel(data_root, annotations, config_path=config)
    assert len(dataset) == 1
    item = dataset[0]
    assert item[CommonKeys.images].shape == (1, 3, 720, 1280)
    assert (
        len(item[CommonKeys.boxes2d])
        == len(item[CommonKeys.boxes2d_classes])
        == len(item[CommonKeys.boxes2d_track_ids])
        == 10
    )
    assert torch.isclose(
        item[CommonKeys.boxes2d_classes],
        torch.tensor([8, 8, 8, 8, 8, 9, 2, 2, 2, 2], dtype=torch.long),
    ).all()
    assert torch.isclose(
        item[CommonKeys.boxes2d_track_ids],
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
    ).all()


def test_3d_data():
    """Test len / getitem methods of scalabel with 3D KITTI data."""
    data_root = get_test_file("track/kitti-samples/", rel_path="run")
    annotations = get_test_file(
        "track/kitti-samples/labels/tracking_training.json", rel_path="run"
    )
    dataset = Scalabel(
        data_root, annotations, targets_to_load=(CommonKeys.boxes3d,)
    )
    assert len(dataset) == 4
    item = dataset[0]
    assert len(item[CommonKeys.boxes3d]) == 7
    assert torch.isclose(
        item[CommonKeys.boxes3d_classes],
        torch.tensor([2, 2, 2, 2, 2, 2, 2], dtype=torch.long),
    ).all()
    assert item[CommonKeys.original_hw] == (375, 1242)

    assert torch.isclose(
        item[CommonKeys.boxes3d_track_ids],
        torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long),
    ).all()
