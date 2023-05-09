# pylint: disable=no-member
"""Test Crop transform."""
import copy
import unittest

import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.crop import (
    CropBoxes2D,
    CropImage,
    CropSegMasks,
    GenCropParameters,
)


class TestCrop(unittest.TestCase):
    """Test Crop transform."""

    data = {
        K.images: np.random.rand(1, 16, 16, 3),
        K.input_hw: (16, 16),
        K.boxes2d: np.array(
            [
                [1.0, 1.0, 4.0, 4.0],
                [4.0, 4.0, 8.0, 8.0],
                [8.0, 8.0, 11.0, 11.0],
            ]
        ),
        K.boxes2d_classes: np.array([1, 2, 3]),
        K.boxes2d_track_ids: np.array([1, 2, 3]),
        K.seg_masks: np.arange(16 * 16).reshape(16, 16),
    }

    def test_crop_image(self):
        """Test the CropImage transform."""
        gen_param = GenCropParameters(shape=(8, 8))
        data = gen_param.apply_to_data(copy.deepcopy(self.data))
        assert "transforms" in data and "crop" in data["transforms"]
        assert (
            "crop_box" in data["transforms"]["crop"]
            and "keep_mask" in data["transforms"]["crop"]
        )
        x1, y1, x2, y2 = data["transforms"]["crop"]["crop_box"]
        assert x2 - x1 == y2 - y1 == 8
        transform = CropImage()
        data = transform.apply_to_data(data)
        self.assertEqual(data["images"].shape, (1, 8, 8, 3))
        self.assertEqual(data["input_hw"], (8, 8))
        assert np.isclose(
            self.data["images"][:, y1:y2, x1:x2, :], data["images"]
        ).all()

    def test_crop_boxes2d(self):
        """Test the CropBoxes2D transform."""
        gen_param = GenCropParameters(shape=(14, 14))
        data = gen_param.apply_to_data(copy.deepcopy(self.data))
        transform = CropBoxes2D()
        data = transform.apply_to_data(data)
        crop_box = data["transforms"]["crop"]["crop_box"]
        keep_mask = data["transforms"]["crop"]["keep_mask"]
        assert (
            len(data["boxes2d"])
            == len(data["boxes2d_classes"])
            == len(data["boxes2d_track_ids"])
            == keep_mask.sum()
        )
        # check overlap of bounding boxes with crop box
        assert (
            ~np.logical_or.reduce(
                (
                    self.data["boxes2d"][:, 0] >= crop_box[2],
                    self.data["boxes2d"][:, 2] <= crop_box[0],
                    self.data["boxes2d"][:, 1] >= crop_box[3],
                    self.data["boxes2d"][:, 3] <= crop_box[1],
                )
            )
            == keep_mask
        ).all()
        # recover original boxes
        x1, y1 = crop_box[:2]
        assert (
            data["boxes2d"] + np.array([x1, y1, x1, y1])
            == self.data["boxes2d"]
        ).all()

    def test_crop_seg_masks(self):
        """Test the CropSegMasks transform."""
        gen_param = GenCropParameters(shape=(14, 14))
        data = gen_param.apply_to_data(copy.deepcopy(self.data))
        transform = CropSegMasks()
        data = transform.apply_to_data(data)
        self.assertEqual(data[K.seg_masks].shape, (14, 14))
