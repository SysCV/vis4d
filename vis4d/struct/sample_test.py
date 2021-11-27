"""Test cases for Vis4D Sample data structures."""
import unittest

import torch

from . import Boxes2D, LabelInstances


class TestLabelInstances(unittest.TestCase):
    """Test cases Vis4D LabelInstances."""

    def test_empty_label(self) -> None:
        """Testcases for empty attribute."""
        empty_labels = LabelInstances()
        self.assertTrue(empty_labels.empty)
        empty_labels.boxes2d = [Boxes2D(torch.zeros(1, 5))]
        self.assertFalse(empty_labels.empty)

    def test_len(self) -> None:
        """Testcases for len."""
        empty_labels = LabelInstances()
        self.assertEqual(len(empty_labels), 1)
        empty_labels = LabelInstances(default_len=2)
        self.assertEqual(len(empty_labels), 2)

    def test_device(self) -> None:
        """Testcases for device."""
        empty_labels = LabelInstances()
        self.assertEqual(empty_labels.device, torch.device("cpu"))

    def test_cat(self) -> None:
        """Testcases for cat."""
        empty_labels1 = LabelInstances()
        empty_labels2 = LabelInstances()
        cat_label = LabelInstances.cat([empty_labels1, empty_labels2])
        self.assertTrue(cat_label.empty)
