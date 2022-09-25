"""Testcases for pointnet."""
import unittest
import torch
from .pointnet import LinearTransformStn


class TestPointnet(unittest.TestCase):
    """Testcase for Pointnet."""

    def setUp(self):
        """Sets up the Test."""
        self.batch_size_ = 8
        self.n_pts_ = 1024

    def test_transform(self) -> None:
        """Tests the learnable features transform."""
        for n_features in [3, 256, 1028]:
            pts = torch.rand(self.batch_size_, n_features, self.n_pts_)
            layer = LinearTransformStn(in_dimension=n_features)
            out = layer(pts)

            self.assertEqual(out.shape[0], self.batch_size_)
            self.assertEqual(out.shape[1], n_features)
            self.assertEqual(out.shape[2], n_features)
