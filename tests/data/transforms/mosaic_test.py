# pylint: disable=no-member,unexpected-keyword-arg
"""Mosaic transformation tests."""
import copy
import unittest
from PIL import Image

import numpy as np
import torch

from tests.util import get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.mosaic import GenerateMosaicParameters, MosaicImages, MosaicBoxes2D


class TestMosaic(unittest.TestCase):
    """Test Mosaic transformation."""

    test_image = np.asarray(Image.open(get_test_file("image.jpg")))[None, ...]

    def test_mosaic_images(self) -> None:
        """Test MosaicImages transformation."""
        data = {
            K.images: copy.deepcopy(self.test_image),
            K.input_hw: [self.test_image.shape[1], self.test_image.shape[2]],
        }
        out_shape = (400, 500)
        params = GenerateMosaicParameters(
            out_shape=out_shape, center_ratio_range=(0.6, 0.6)
        )
        transform = MosaicImages()
        data = params.apply_to_data([data] * 4)
        data = transform.apply_to_data(data)[0]
        self.assertEqual(
            data[K.images].shape, (1, out_shape[0] * 2, out_shape[1] * 2, 3)
        )
        self.assertEqual(
            data[K.input_hw], (out_shape[0] * 2, out_shape[1] * 2)
        )
        assert "transforms" in data
        assert "mosaic" in data["transforms"]
        assert len(data["transforms"]["mosaic"]["paste_coords"]) == 4
        assert len(data["transforms"]["mosaic"]["crop_coords"]) == 4
        assert len(data["transforms"]["mosaic"]["im_shapes"]) == 4
        assert len(data["transforms"]["mosaic"]["im_scales"]) == 4
        assert np.allclose(
            data[K.images],
            torch.load(get_test_file("mosaic_images.npy")),
            atol=1e-4,
        )
    
    def test_mosaic_boxes2d(self) -> None:
        """Test MosaicBoxes2D transformation."""
        data = {
            K.images: copy.deepcopy(self.test_image),
            K.input_hw: [self.test_image.shape[1], self.test_image.shape[2]],
            K.boxes2d: np.array(
                [
                    [100.0, 100.0, 200.0, 200.0],
                    [200.0, 100.0, 300.0, 200.0],
                    [0.0, 0.0, 100.0, 100.0],
                ]
            ),
            K.boxes2d_classes: np.array([1, 2, 3]),
            K.boxes2d_track_ids: np.array([1, 2, 3]),
        }

        params = GenerateMosaicParameters(
            out_shape=(400, 500), center_ratio_range=(0.6, 0.6)
        )
        transform = MosaicBoxes2D()
        data = params.apply_to_data([copy.deepcopy(data) for _ in range(4)])
        data = MosaicImages().apply_to_data(data)
        data = transform.apply_to_data(data)[0]
        box_data = [data[K.boxes2d], data[K.boxes2d_classes], data[K.boxes2d_track_ids]]
        for pred, gt in zip(box_data, torch.load(get_test_file("mosaic_boxes2d.npy"))):
            assert np.allclose(pred, gt, atol=1e-4)
