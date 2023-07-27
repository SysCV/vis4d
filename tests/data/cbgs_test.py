"""Class-balanced Grouping and Sampling for 3D Object Detection test."""
from tests.util import get_test_data
from vis4d.data.cbgs import CBGSDataset
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_class_map


def test_cbgs():
    """Test CBGS dataset."""
    data_root = get_test_data("nuscenes_test", absolute_path=False)

    nusc = NuScenes(
        data_root=data_root,
        version="v1.0-mini",
        split="mini_train",
        cache_as_binary=True,
        cached_file_path=f"{data_root}/mini_train.pkl",
    )

    cbgs_dataset = CBGSDataset(dataset=nusc, class_map=nuscenes_class_map)

    assert len(cbgs_dataset) == 1690
