"""Tests for the Vis4D image visualizer."""
from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from vis4d.unittest.util import get_test_file

from .semantic_mask_visualizer import SemanticMaskVisualizer

SEM_MAPPING = {  # pylint:disable=duplicate-code
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
        with open(get_test_file("mask_data.pkl"), "rb") as f:
            testcase_in = pickle.load(f)

            self.images = [e["img"] for e in testcase_in]
            self.masks = [np.stack(e["masks"]) for e in testcase_in]
            self.class_ids = [np.stack(e["class_id"]) for e in testcase_in]

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
        vis = SemanticMaskVisualizer(n_colors=20, class_id_mapping=SEM_MAPPING)

        vis.process_single_image(
            self.images[0], self.masks[0], self.class_ids[0]
        )
        vis.save_to_disk(self.test_dir)

        self.assert_img_equal(
            os.path.join(self.test_dir, "0000.png"),
            get_test_file("mask_result.png"),
        )
