"""Class-balanced Grouping and Sampling for 3D Object Detection test."""
from tests.util import get_test_data
from vis4d.data.cbgs import CBGSDataset
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_class_map


def test_cbgs():
    """Test CBGS dataset."""
    nusc = NuScenes(
        data_root=get_test_data("nuscenes_test"),
        version="v1.0-mini",
        split="mini_train",
    )

    cbgs_dataset = CBGSDataset(dataset=nusc, class_map=nuscenes_class_map)

    assert len(cbgs_dataset) == 1690
