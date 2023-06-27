"""Tests for the semantic mask visualizer."""
import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from tests.util import get_test_file
from vis4d.vis.image.seg_mask_visualizer import SegMaskVisualizer

from .util import COCO_COLOR_MAPPING


class TestSegMaskVis(unittest.TestCase):
    """Testcase for Segmentation Mask Visualizer."""

    def setUp(self) -> None:
        """Creates a tmp directory and loads input data."""
        self.test_dir = tempfile.mkdtemp()
        with open(get_test_file("mask_data.pkl"), "rb") as f:
            testcase_in = pickle.load(f)

            self.images = [e["img"] for e in testcase_in]
            self.image_names = ["0000", "0001"]
            self.masks = [np.stack(e["masks"]) for e in testcase_in]
            self.class_ids = [np.stack(e["class_id"]) for e in testcase_in]

        self.vis = SegMaskVisualizer(
            n_colors=100, class_id_mapping=COCO_COLOR_MAPPING, vis_freq=1
        )

    def tearDown(self) -> None:
        """Removes the tmp directory after the test."""
        shutil.rmtree(self.test_dir)

    def assert_img_equal(self, pred_path: str, gt_path: str) -> None:
        """Compares two images.

        Args:
            pred_path (str): Path to predicted image
            gt_path (str): Path to groundtruth image
        """
        pred_np = np.asarray(Image.open(pred_path))
        gt_np = np.asarray(Image.open(gt_path))
        self.assertTrue(np.allclose(pred_np, gt_np))

    def test_single_mask_vis(self) -> None:
        """Tests visualization of seg masks."""
        self.vis.process(
            0,
            [self.images[0]],
            [self.image_names[0]],
            [self.masks[0]],
            [self.class_ids[0]],
        )
        self.vis.save_to_disk(cur_iter=1, output_folder=self.test_dir)

        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("ins_mask_result.png"),
        )
        self.vis.reset()

        # test mask input with shape [H, W]
        hw_mask = np.full(self.masks[0].shape[1:], 255)
        for mask, cls_id in zip(self.masks[0], self.class_ids[0]):
            hw_mask[mask > 0] = cls_id
        self.vis.process(
            0,
            [self.images[0]],
            [self.image_names[0]],
            [hw_mask],
        )
        self.vis.save_to_disk(cur_iter=1, output_folder=self.test_dir)

        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("sem_mask_result.png"),
        )
        self.vis.reset()
