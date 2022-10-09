"""Testcases for pointnet."""
import unittest

import torch

from vis4d.model.segment3d.pointnet import PointnetSegmentationModel


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
        segmenter = PointnetSegmentationModel(num_classes=n_classes)

        # Inference
        pts = torch.rand(self.batch_size, n_features, self.n_pts)
        out = segmenter(pts)
        self.assertEqual(type(out), torch.Tensor)
        self.assertEqual(tuple(out.shape), (self.batch_size, self.n_pts))

    def test_train(self) -> None:
        """Tests the forward and training path of the segmentation network."""
        n_features = 3
        n_classes = 10
        segmenter = PointnetSegmentationModel(num_classes=n_classes)

        # Training
        pts = torch.rand(self.batch_size, n_features, self.n_pts)
        targets = torch.randint(n_classes, (self.batch_size, self.n_pts))
        out = segmenter(pts, targets)

        # Check prediction size matches
        self.assertEqual(type(out), tuple)
        self.assertEqual(tuple(out[0].shape), (self.batch_size, self.n_pts))
        # Check that losses are well defined
        for l in out[1]:
            self.assertEqual(type(l), torch.Tensor)
