"""Testcases for pointnet."""
import unittest

import torch

from .pointnet import LinearTransform, PointNetEncoder, PointNetEncoderOut


class TestPointnet(unittest.TestCase):
    """Testcase for Pointnet."""

    def setUp(self):
        """Sets up the Test."""
        self.batch_size_ = 8
        self.n_pts_ = 1024

    def test_transform_shapes(self) -> None:
        """Tests the shapes of the learnable features transform."""
        for n_features in [3, 256, 1028]:
            pts = torch.rand(self.batch_size_, n_features, self.n_pts_)
            layer = LinearTransform(in_dimension=n_features)
            out = layer(pts)

            self.assertEqual(out.shape[0], self.batch_size_)
            self.assertEqual(out.shape[1], n_features)
            self.assertEqual(out.shape[2], n_features)

    def test_encoder_shape(self) -> None:
        """Tests the shapes of the full pointnet encoder."""
        for n_features in [3, 6, 9]:
            for out_dim in [1024, 2048]:
                mlp_dimensions = [[64, 64], [64, 128]]
                pts = torch.rand(self.batch_size_, n_features, self.n_pts_)
                encoder = PointNetEncoder(
                    in_dimensions=n_features, out_dimensions=out_dim
                )
                out = encoder(pts)

                self.assertEqual(out.features.shape[0], self.batch_size_)
                self.assertEqual(out.features.shape[1], out_dim)

                self.assertEqual(len(out.transformations), len(mlp_dimensions))
