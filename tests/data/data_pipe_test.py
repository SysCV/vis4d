"""Test data pipelines."""

import os

from tests.util import get_test_data
from vis4d.data.data_pipe import DataPipe, MultiSampleDataPipe
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.transforms.mosaic import GenMosaicParameters, MosaicImages


def test_data_pipe():
    """Test DataPipe."""
    dataset = BDD100K(
        data_root=os.path.join(get_test_data("bdd100k_test"), "track/images"),
        annotation_path=os.path.join(
            get_test_data("bdd100k_test"), "track/labels"
        ),
        config_path="box_track",
    )
    datapipe = DataPipe(dataset)

    batch = datapipe[0]
    assert set(batch.keys()) == {
        "images",
        "input_hw",
        "original_images",
        "original_hw",
        "axis_mode",
        "frame_ids",
        "sample_names",
        "sequence_names",
        "boxes2d",
        "boxes2d_classes",
        "boxes2d_track_ids",
    }

    assert datapipe[0]["frame_ids"] - datapipe[1]["frame_ids"] in [1, -1]


def test_mosaic_data_pipe():
    """Test MultiSampleDataPipe."""
    dataset = BDD100K(
        data_root=os.path.join(get_test_data("bdd100k_test"), "track/images"),
        annotation_path=os.path.join(
            get_test_data("bdd100k_test"), "track/labels"
        ),
        config_path="box_track",
    )
    datapipe = MultiSampleDataPipe(
        dataset, [[GenMosaicParameters((128, 128)), MosaicImages()]]
    )

    batch = datapipe[0]
    assert set(batch.keys()) == {
        "images",
        "input_hw",
        "original_images",
        "original_hw",
        "axis_mode",
        "frame_ids",
        "sample_names",
        "sequence_names",
        "boxes2d",
        "boxes2d_classes",
        "boxes2d_track_ids",
        "transforms",
    }
    assert "mosaic" in batch["transforms"]

    assert datapipe[0]["frame_ids"] - datapipe[1]["frame_ids"] in [1, -1]
