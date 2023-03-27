"""S3DIS dataset testing class."""
import unittest

import numpy as np

from tests.util import get_test_data, isclose_on_all_indices_numpy
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.shift import SHIFT
from vis4d.data.io import HDF5Backend, ZipBackend

IMAGE_INDICES = np.array([0, 419, 581117, 1023997])
IMAGE_VALUES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [15.0, 14.0, 12.0],
        [4.0, 12.0, 0.0],
    ]
)

INSTANCE_MASK_INDICES = np.array([0, 419, 521576, 1557200])
INSTANCE_MASK_VALUES = np.array([0, 0, 1, 1])

SEMANTIC_MASK_INDICES = np.array([0, 419, 581117, 1023997])
SEMANTIC_MASK_VALUES = np.array([13, 5, 14, 22])

DEPTH_MAP_INDICES = np.array([0, 419, 581117, 1023997])
DEPTH_MAP_VALUES = np.array([16777.2109, 134.4130, 320.8880, 42.8810])

OPTICAL_FLOW_INDICES = np.array([0, 419, 581117, 1023997])
OPTICAL_FLOW_VALUES = np.array(
    [
        [-0.0010, -0.0010],
        [-0.0010, -0.0010],
        [-0.0010, -0.0010],
        [-0.0010, -0.0010],
    ]
)

POINTS3D_INDICES = np.array([0, 100, 5000, 51110])
POINTS3D_VALUES = np.array(
    [
        [-116.9319, -14.1797, 20.7693, 0.6198],
        [26.7530, -19.5690, 5.8446, 0.8740],
        [8.3980, -8.7958, 1.6308, 0.9521],
        [-0.3214, 0.0103, -0.0567, 0.9987],
    ]
)


class SHIFTTest(unittest.TestCase):
    """Test SHIFT dataloading."""

    dataset = SHIFT(data_root=get_test_data("shift_test"), split="val")

    dataset_zip = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=[
            K.images,
            K.input_hw,
            K.intrinsics,
            K.boxes2d,
            K.segmentation_masks,
            K.depth_maps,
            K.optical_flows,
        ],
        backend=ZipBackend(),
    )

    dataset_multiview = SHIFT(
        data_root=get_test_data("shift_test"),
        split="val",
        keys_to_load=[
            K.images,
            K.input_hw,
            K.intrinsics,
            K.boxes2d,
            K.boxes2d_classes,
            K.boxes2d_track_ids,
            K.boxes3d,
            K.instance_masks,
            K.segmentation_masks,
            K.depth_maps,
            K.optical_flows,
            K.points3d,
        ],
        views_to_load=["front", "left_90", "center"],
        backend=HDF5Backend(),
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 1)

    def test_len_zip(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset_zip), 1)

    def test_len_multiview(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset_multiview), 1)

    def test_video_indices(self) -> None:
        """Test if video indices are correct."""
        self.assertEqual(self.dataset.video_to_indices, {"007b-4e72": [0]})

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        for view in ("front", "left_90"):
            self.assertEqual(
                tuple(self.dataset_multiview[0][view].keys()),
                (
                    K.images,
                    K.input_hw,
                    K.intrinsics,
                    K.boxes2d,
                    K.boxes2d_classes,
                    K.boxes2d_track_ids,
                    K.boxes3d,
                    K.instance_masks,
                    K.segmentation_masks,
                    K.depth_maps,
                    K.optical_flows,
                ),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.images].shape,
                (1, 800, 1280, 3),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.instance_masks].shape,
                (2, 800, 1280) if view == "front" else (0, 0, 0),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.segmentation_masks].shape,
                (800, 1280),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.depth_maps].shape,
                (800, 1280),
            )
            self.assertEqual(
                self.dataset_multiview[0][view][K.optical_flows].shape,
                (800, 1280, 2),
            )

        item = self.dataset_multiview[0]["front"]
        assert isclose_on_all_indices_numpy(
            item[K.images].reshape(-1, 3),
            IMAGE_INDICES,
            IMAGE_VALUES,
        )
        assert isclose_on_all_indices_numpy(
            item[K.instance_masks].reshape(-1),
            INSTANCE_MASK_INDICES,
            INSTANCE_MASK_VALUES,
        )
        assert isclose_on_all_indices_numpy(
            item[K.segmentation_masks].reshape(-1),
            SEMANTIC_MASK_INDICES,
            SEMANTIC_MASK_VALUES,
        )
        assert isclose_on_all_indices_numpy(
            item[K.depth_maps].reshape(-1),
            DEPTH_MAP_INDICES,
            DEPTH_MAP_VALUES,
        )
        assert isclose_on_all_indices_numpy(
            item[K.optical_flows].reshape(-1, 2),
            OPTICAL_FLOW_INDICES,
            OPTICAL_FLOW_VALUES,
        )

        for view in ("center",):
            self.assertEqual(
                tuple(self.dataset_multiview[0][view].keys()),
                (K.boxes3d, K.points3d),
            )
            item = self.dataset_multiview[0][view][K.points3d]
            self.assertEqual(item.shape, (51111, 4))
            assert isclose_on_all_indices_numpy(
                item,
                POINTS3D_INDICES,
                POINTS3D_VALUES,
            )
