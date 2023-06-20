"""Test DataPipe."""
import os

from tests.util import get_test_data
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets import BDD100K


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
