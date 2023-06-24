# pylint: disable=no-member,unexpected-keyword-arg
"""Affine transformation tests."""
import copy
import unittest

import numpy as np
from PIL import Image

from tests.util import get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.affine import (
    AffineBoxes2D,
    AffineImages,
    GenAffineParameters,
)


class TestAffine(unittest.TestCase):
    """Test Affine transformation."""

    test_image = np.asarray(Image.open(get_test_file("image.jpg")))[None, ...]

    def test_affine_images(self) -> None:
        """Test AffineImages transformation."""
        data = {
            K.images: copy.deepcopy(self.test_image),
            K.input_hw: [self.test_image.shape[1], self.test_image.shape[2]],
        }
        params = GenAffineParameters()
        transform = AffineImages()
        data = params.apply_to_data([data])
        data = transform.apply_to_data(data)[0]
        assert "transforms" in data
        assert "affine" in data["transforms"]
        assert data["transforms"]["affine"]["warp_matrix"].shape == (3, 3)
        assert data["transforms"]["affine"]["height"] == 230
        assert data["transforms"]["affine"]["width"] == 352

    def test_affine_boxes2d(self) -> None:
        """Test AffineBoxes2D transformation."""
        data = {
            K.images: copy.deepcopy(self.test_image),
            K.input_hw: [self.test_image.shape[1], self.test_image.shape[2]],
            K.boxes2d: np.array(
                [
                    [100.0, 100.0, 200.0, 200.0],
                    [200.0, 100.0, 300.0, 200.0],
                    [0.0, 0.0, 100.0, 100.0],
                ],
                dtype=np.float32,
            ),
            K.boxes2d_classes: np.array([1, 2, 3]),
            K.boxes2d_track_ids: np.array([1, 2, 3]),
        }

        params = GenAffineParameters()
        transform = AffineBoxes2D()
        data = params.apply_to_data([data])
        data = AffineImages().apply_to_data(data)
        data = transform.apply_to_data(data)[0]
        box_data = [
            data[K.boxes2d],
            data[K.boxes2d_classes],
            data[K.boxes2d_track_ids],
        ]
        assert box_data[0].shape == (3, 4)
        assert np.allclose(box_data[1], np.array([1, 2, 3]))
        assert np.allclose(box_data[2], np.array([1, 2, 3]))
