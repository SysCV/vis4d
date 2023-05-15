"""Tests for the bounding box visualizer."""
import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from tests.util import get_test_file
from vis4d.common.typing import NDArrayF64, NDArrayI64
from vis4d.vis.image.bounding_box_visualizer import BoundingBoxVisualizer

from .util import COLOR_MAPPING


class TestBoundingBoxVis(unittest.TestCase):
    """Testcase for Bounding Box Visualizer."""

    def setUp(self) -> None:
        """Creates a tmp directory and loads input data."""
        self.test_dir = tempfile.mkdtemp()
        with open(get_test_file("draw_bbox_with_cts.pkl"), "rb") as f:
            testcase_gt = pickle.load(f)

        self.images: list[NDArrayF64] = testcase_gt["imgs"]
        self.image_names: list[str] = ["0000", "0001"]
        self.boxes: list[NDArrayF64] = testcase_gt["boxes"]
        self.classes: list[NDArrayI64] = testcase_gt["classes"]
        self.scores: list[NDArrayF64] = testcase_gt["scores"]
        self.tracks = [np.arange(len(b)) for b in self.boxes]

        self.vis = BoundingBoxVisualizer(
            n_colors=20, class_id_mapping=COLOR_MAPPING, vis_freq=1
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

    def test_single_bbox_vis(self) -> None:
        """Tests visualization of single boudning boxes."""
        self.vis.process_single_image(
            image=self.images[0],
            image_name=self.image_names[0],
            boxes=self.boxes[0],
            scores=self.scores[0],
            class_ids=self.classes[0],
            track_ids=self.tracks[0],
        )

        self.vis.save_to_disk(cur_iter=1, output_folder=self.test_dir)
        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("bbox_with_cts_target.png"),
        )
        self.vis.reset()

    def test_single_bbox_vis_no_tracks(self) -> None:
        """Tests visualization of single bounding boxes without track ids."""
        self.vis.process_single_image(
            image=self.images[0],
            image_name=self.image_names[0],
            boxes=self.boxes[0],
            scores=self.scores[0],
            class_ids=self.classes[0],
            track_ids=None,
        )

        self.vis.save_to_disk(cur_iter=1, output_folder=self.test_dir)
        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("bbox_with_cs_target.png"),
        )
        self.vis.reset()

    def test_single_bbox_vis_only_class(self) -> None:
        """Tests visualization of single bounding boxes with only classes."""
        self.vis.process_single_image(
            image=self.images[0],
            image_name=self.image_names[0],
            boxes=self.boxes[0],
            scores=None,
            class_ids=self.classes[0],
            track_ids=None,
        )

        self.vis.save_to_disk(cur_iter=1, output_folder=self.test_dir)
        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("bbox_with_c_target.png"),
        )
        self.vis.reset()

    def test_batched_vis(self) -> None:
        """Test visualization of bboxes with multiple images."""
        self.vis.process(
            cur_iter=1,
            images=self.images,
            image_names=self.image_names,
            boxes=self.boxes,
            scores=self.scores,
            class_ids=self.classes,
            track_ids=self.tracks,
        )

        self.vis.save_to_disk(cur_iter=1, output_folder=self.test_dir)
        for i in range(2):
            self.assert_img_equal(
                os.path.join(self.test_dir, f"000{i}.png"),
                get_test_file(f"bbox_batched_{i}.png"),
            )
        self.vis.reset()
