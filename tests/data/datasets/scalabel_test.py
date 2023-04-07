"""Tests for scalabel dataset."""
import os

import numpy as np

from tests.util import get_test_data, isclose_on_all_indices_numpy
from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.scalabel import Scalabel

if SCALABEL_AVAILABLE:
    from scalabel.label.typing import Config

IMAGE_INDICES = np.array([0, 1, 232875, 465749])
IMAGE_VALUES = np.array(
    [
        [173.0, 255.0, 255.0],
        [173.0, 255.0, 255.0],
        [40.0, 96.0, 86.0],
        [14.0, 16.0, 20.0],
    ]
)


def test_setup_categories():
    """Test setup categories."""
    data_root = get_test_data("bdd100k_test")
    annotations = os.path.join(data_root, "detect/labels/annotation.json")
    dataset = Scalabel(
        data_root,
        annotations,
        category_map={K.images: {}, K.boxes2d: {"car": 0, "person": 1}},
        config_path=Config(categories=[]),
    )
    assert isinstance(dataset.cats_name2id, dict)
    assert dataset.cats_name2id[K.images] == {}
    assert dataset.cats_name2id[K.boxes2d] == {"car": 0, "person": 1}


def test_3d_data() -> None:
    """Test 3D bounding box data from scalabel."""
    data_root = get_test_data("kitti_test")
    annotations = os.path.join(data_root, "labels/tracking_training.json")
    dataset = Scalabel(
        data_root,
        annotations,
        keys_to_load=(
            K.images,
            K.original_hw,
            K.boxes3d,
            K.boxes3d_classes,
            K.boxes3d_track_ids,
        ),
    )
    assert len(dataset) == 4

    item = dataset[0]

    assert item[K.images].shape == (1, 375, 1242, 3)
    assert item[K.original_hw] == (375, 1242)
    assert isclose_on_all_indices_numpy(
        item[K.images].astype(np.float32).reshape(-1, 3),
        IMAGE_INDICES,
        IMAGE_VALUES,
    )

    assert len(item[K.boxes3d]) == 7
    assert np.isclose(
        item[K.boxes3d][0],
        np.array(
            [
                2.921483,
                0.755883,
                6.348542,
                1.509920,
                1.850000,
                4.930564,
                0.707107,
                0.000000,
                -0.707107,
                0.000000,
            ],
            dtype=np.float32,
        ),
    ).all()
    assert np.isclose(
        item[K.boxes3d_classes],
        np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.int64),
    ).all()
    assert item[K.original_hw] == (375, 1242)

    assert np.isclose(
        item[K.boxes3d_track_ids],
        np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    ).all()
