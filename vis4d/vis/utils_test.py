"""Visualization utils test cases."""
import math
import unittest

import numpy as np
import torch
from PIL import Image

from vis4d.struct_to_revise import Intrinsics
from vis4d.unittest.utils import generate_dets, generate_instance_masks

from .utils import (
    box3d_to_corners,
    generate_colors,
    preprocess_boxes,
    preprocess_image,
    preprocess_intrinsics,
    preprocess_masks,
)


class TestUtils(unittest.TestCase):
    """Testcase class vis utils."""

    def test_generate_colors(self) -> None:
        """Test generate colors method."""
        cols = generate_colors(10)
        self.assertEqual(len(cols), 10)
        for col in cols:
            self.assertEqual(len(col), 3)
            for val in col:
                self.assertTrue(isinstance(val, int))
                self.assertTrue(0 <= val < 256)

    def test_preprocess_image(self) -> None:
        """Test preprocess_image method."""
        img = torch.rand((3, 128, 128))
        proc_img = preprocess_image(img, mode="BGR")
        self.assertTrue(isinstance(proc_img, Image.Image))
        proc_img = np.array(proc_img)
        self.assertEqual(proc_img.shape, (128, 128, 3))
        self.assertTrue(np.min(proc_img) >= 0)
        self.assertTrue(np.max(proc_img) < 256)

    def test_preprocess_intrinsics(self) -> None:
        """Test preprocess_intrinsics method."""
        mat1 = preprocess_intrinsics(np.eye(3))
        mat2 = preprocess_intrinsics(Intrinsics(torch.eye(3)))
        self.assertTrue(np.isclose(mat1, mat2).all())

    def test_preprocess_boxes(self) -> None:
        """Test preprocess_boxes method."""
        dets = [generate_dets(128, 128, 10, track_ids=True)]
        dets[0].track_ids = dets[0].track_ids.unsqueeze(-1)

        # with score
        proc_dets, cols, labels = preprocess_boxes(dets)
        self.assertTrue(
            len(dets[0]) == len(proc_dets) == len(cols) == len(labels)
        )
        self.assertTrue(len(cols) == len(set(cols)))
        for det, proc_det, label in zip(dets[0], proc_dets, labels):
            det_box = det.boxes[0, :4].numpy().tolist()
            self.assertEqual(det_box, proc_det)
            self.assertEqual(0, int(label[2]))

        # without score
        dets[0].boxes = dets[0].boxes[:, :4]
        proc_dets, cols, labels = preprocess_boxes(dets)
        self.assertTrue(
            len(dets[0]) == len(proc_dets) == len(cols) == len(labels)
        )
        self.assertTrue(len(cols) == len(set(cols)))
        for det, proc_det, label in zip(dets[0], proc_dets, labels):
            det_box = det.boxes[0, :4].numpy().tolist()
            self.assertEqual(det_box, proc_det)
            self.assertEqual(0, int(label[2]))

        dets[0].track_ids = None
        proc_dets, cols, _ = preprocess_boxes(dets)
        self.assertTrue(len(set(cols)) == 1)

        dets[0].class_ids = None
        proc_dets, cols, _ = preprocess_boxes(dets)
        self.assertTrue(len(set(cols)) == 1)

    def test_preprocess_masks(self) -> None:
        """Test preprocess_masks method."""
        masks = [generate_instance_masks(128, 128, 10, track_ids=True)]
        masks[0].track_ids = masks[0].track_ids.unsqueeze(-1)

        # with score
        proc_masks, cols = preprocess_masks(masks)
        self.assertTrue(len(masks[0]) == len(proc_masks) == len(cols))
        self.assertTrue(len(cols) == len(set(cols)))
        for mask, proc_mask in zip(masks[0], proc_masks):
            self.assertTrue((mask.masks.numpy() == proc_mask / 255).all())

        masks[0].track_ids = None
        proc_masks, cols = preprocess_masks(masks)
        self.assertTrue(len(set(cols)) == 1)

        masks[0].class_ids = None
        proc_masks, cols = preprocess_masks(masks)
        self.assertTrue(len(set(cols)) == 1)

    def test_box3d_to_corners(self) -> None:
        """Test for box3d_to_corners function."""
        box_corners = [
            [11.0, 11.0, 8.0],
            [9.0, 11.0, 8.0],
            [9.0, 11.0, 12.0],
            [11.0, 11.0, 12.0],
            [11.0, 9.0, 8.0],
            [9.0, 9.0, 8.0],
            [9.0, 9.0, 12.0],
            [11.0, 9.0, 12.0],
        ]
        corners = box3d_to_corners([10, 10, 10, 2, 2, 4, 0, math.pi / 2, 0])
        self.assertTrue((corners == box_corners).all())
