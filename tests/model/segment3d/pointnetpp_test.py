"""Testcases for pointnet."""
import unittest

import torch

from vis4d.data.const import CommonKeys
from vis4d.model.segment3d.pointnetpp import PointNet2SegmentationModel


class TestPointnet(unittest.TestCase):
    """Testcase for Pointnet."""

    def setUp(self):
        """Sets up the Test."""
        self.batch_size = 8
        self.n_pts = 1024

    def test_inference(self) -> None:
        """Tests the forward and training path of the segmentation network."""
        n_features = 3
        n_classes = 10
        segmenter = PointNet2SegmentationModel(
            num_classes=n_classes, in_dimensions=n_features
        )

        # Inference
        pts = torch.rand(self.batch_size, n_features, self.n_pts)
        out = segmenter(pts)
        self.assertEqual(
            tuple(out[CommonKeys.semantics3d].shape),
            (self.batch_size, self.n_pts),
        )

    def test_train(self) -> None:
        """Tests the forward and training path of the segmentation network."""
        n_features = 3
        n_classes = 10
        segmenter = PointNet2SegmentationModel(
            num_classes=n_classes, in_dimensions=n_features
        )

        # Training
        pts = torch.rand(self.batch_size, n_features, self.n_pts)
        targets = torch.randint(n_classes, (self.batch_size, self.n_pts))
        out = segmenter(pts, targets)

        # Check prediction size matches
        self.assertEqual(
            tuple(out.class_logits.shape),
            (self.batch_size, n_classes, self.n_pts),
        )
