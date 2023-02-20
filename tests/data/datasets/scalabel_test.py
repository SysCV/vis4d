"""Tests for scalabel dataset."""
import os

import torch

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.datasets.scalabel import Scalabel
from vis4d.data.datasets.bdd100k import BDD100K


def test_len_getitem():
    """Test len / getitem methods of scalabel."""
    data_root = get_test_data("bdd100k_test/detect")
    annotations = os.path.join(data_root, "labels")
    config = os.path.join(data_root, "config.toml")
    dataset = Scalabel(
        os.path.join(data_root, "images"),
        annotations,
        keys_to_load=(
            Keys.images,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.boxes2d_track_ids,
        ),
        config_path=config,
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert item[Keys.images].shape == (1, 3, 720, 1280)
    assert (
        len(item[Keys.boxes2d])
        == len(item[Keys.boxes2d_classes])
        == len(item[Keys.boxes2d_track_ids])
        == 10
    )
    assert torch.isclose(
        item[Keys.boxes2d_classes],
        torch.tensor([8, 8, 8, 8, 8, 9, 2, 2, 2, 2], dtype=torch.long),
    ).all()
    assert torch.isclose(
        item[Keys.boxes2d_track_ids],
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
    ).all()


def test_3d_data():
    """Test 3D bounding box data from scalabel."""
    data_root = get_test_data("kitti_test")
    annotations = os.path.join(data_root, "labels/tracking_training.json")
    dataset = Scalabel(
        data_root,
        annotations,
        keys_to_load=(
            Keys.images,
            Keys.original_hw,
            Keys.boxes3d,
            Keys.boxes3d_classes,
            Keys.boxes3d_track_ids,
        ),
    )
    assert len(dataset) == 4
    item = dataset[0]
    assert len(item[Keys.boxes3d]) == 7
    assert torch.isclose(
        item[Keys.boxes3d_classes],
        torch.tensor([2, 2, 2, 2, 2, 2, 2], dtype=torch.long),
    ).all()
    assert item[Keys.original_hw] == (375, 1242)

    assert torch.isclose(
        item[Keys.boxes3d_track_ids],
        torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long),
    ).all()


def test_instance_segmentation():
    """Test instance segmentation annotation from scalabel."""
    data_root = get_test_data("bdd100k_test")
    annotations = os.path.join(data_root, "detect/labels/annotation.json")
    config_path = os.path.join(data_root, "detect/insseg_config.toml")
    dataset = BDD100K(
        data_root,
        annotations,
        keys_to_load=(
            Keys.images,
            Keys.boxes2d,
            Keys.boxes2d_classes,
            Keys.boxes2d_track_ids,
            Keys.masks,
        ),
        config_path=config_path,
    )
    assert len(dataset) == 4
    item = dataset[0]
    assert len(item[Keys.boxes3d]) == 7
    assert torch.isclose(
        item[Keys.boxes3d_classes],
        torch.tensor([2, 2, 2, 2, 2, 2, 2], dtype=torch.long),
    ).all()
    assert item[Keys.original_hw] == (375, 1242)

    assert torch.isclose(
        item[Keys.boxes3d_track_ids],
        torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long),
    ).all()
