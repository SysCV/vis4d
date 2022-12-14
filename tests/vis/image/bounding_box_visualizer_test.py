"""Tests for the Vis4D image visualizer."""
from __future__ import annotations

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

SEM_MAPPING = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


class TestConvOccnet(unittest.TestCase):
    """Testcase for Bounding Box Visualizer."""

    def setUp(self) -> None:
        """Creates a tmp directory and loads input data."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        with open(get_test_file("draw_bbox_with_cts.pkl"), "rb") as f:
            testcase_gt = pickle.load(f)

        self.images: list[NDArrayF64] = testcase_gt["imgs"]
        self.boxes: list[NDArrayF64] = testcase_gt["boxes"]
        self.classes: list[NDArrayI64] = testcase_gt["classes"]
        self.scores: list[NDArrayF64] = testcase_gt["scores"]
        self.tracks = [np.arange(len(b)) for b in self.boxes]

    def tearDown(self) -> None:
        """Removes the tmp directory."""
        # Remove the directory after the test
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

    # Single image visualization
    def test_single_bbox_vis(self) -> None:
        """Tests visualization of bboxes with classes, scores and tracks."""
        vis = BoundingBoxVisualizer(n_colors=20, class_id_mapping=SEM_MAPPING)

        # Single image
        vis.process_single_image(
            self.images[0],
            boxes=self.boxes[0],
            scores=self.scores[0],
            class_ids=self.classes[0],
            track_ids=self.tracks[0],
        )

        vis.save_to_disk(self.test_dir)
        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("bbox_with_cts_target.png"),
        )

    def test_single_bbox_vis_no_tracks(self) -> None:
        """Tests visualization of bboxes with classes and scores."""
        vis = BoundingBoxVisualizer(n_colors=20, class_id_mapping=SEM_MAPPING)
        # Single image
        vis.process_single_image(
            self.images[0],
            boxes=self.boxes[0],
            scores=self.scores[0],
            class_ids=self.classes[0],
            track_ids=None,
        )

        vis.save_to_disk(self.test_dir)
        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("bbox_with_cs_target.png"),
        )

    def test_single_bbox_vis_only_class(self) -> None:
        """Tests visualization of bboxes with only classes."""
        vis = BoundingBoxVisualizer(n_colors=20, class_id_mapping=SEM_MAPPING)
        # Single image
        vis.process_single_image(
            self.images[0],
            boxes=self.boxes[0],
            scores=None,
            class_ids=self.classes[0],
            track_ids=None,
        )
        vis.save_to_disk(self.test_dir)
        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("bbox_with_c_target.png"),
        )

    def test_batched_vis(self) -> None:
        """Test visualization of bboxes with multiple images."""
        vis = BoundingBoxVisualizer(n_colors=20, class_id_mapping=SEM_MAPPING)
        # Single image
        vis.process(
            self.images,
            boxes=self.boxes,
            scores=self.scores,
            class_ids=self.classes,
            track_ids=self.tracks,
        )
        vis.save_to_disk(self.test_dir)

        for i in range(2):
            self.assert_img_equal(
                os.path.join(self.test_dir, f"000{i}.png"),
                get_test_file(f"bbox_batched_{i}.png"),
            )
