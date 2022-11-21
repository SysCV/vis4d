"""Testcases for pointnet."""
import unittest

import torch

from .pointnet import LinearTransform, PointNetEncoder, PointNetSegmentation


class TestPointnet(unittest.TestCase):
    """Testcase for Pointnet."""

    def setUp(self):
        """Sets up the Test."""
        self.batch_size_ = 8
        self.n_pts_ = 1024

    def test_transform_shapes(self) -> None:
        """Tests the shapes of the learnable features transform."""
        for n_features in (3, 256, 1028):
            pts = torch.rand(self.batch_size_, n_features, self.n_pts_)
            layer = LinearTransform(in_dimension=n_features)
            out = layer(pts)

            self.assertEqual(out.shape[0], self.batch_size_)
            self.assertEqual(out.shape[1], n_features)
            self.assertEqual(out.shape[2], n_features)

    def test_encoder_shape(self) -> None:
        """Tests the shapes of the full pointnet encoder."""
        for n_features in (3, 6, 9):
            for out_dim in (1024, 2048):
                mlp_dimensions = [[64, 64], [64, 128]]
                pts = torch.rand(self.batch_size_, n_features, self.n_pts_)
                encoder = PointNetEncoder(
                    in_dimensions=n_features, out_dimensions=out_dim
                )
                out = encoder(pts)

                self.assertEqual(out.features.shape[0], self.batch_size_)
                self.assertEqual(out.features.shape[1], out_dim)

                self.assertEqual(len(out.transformations), len(mlp_dimensions))

    def test_segmentation_shape(self) -> None:
        """Tests the shapes of the segmentation output using pointnet."""
        n_feats = 3
        n_classes = 12
        segmenter = PointNetSegmentation(
            n_classes=n_classes, in_dimensions=n_feats
        )
        pts = torch.rand(self.batch_size_, n_feats, self.n_pts_)
        out = segmenter(pts)
        predict_logits = out.class_logits
        self.assertEqual(
            tuple(predict_logits.shape),
            (self.batch_size_, n_classes, self.n_pts_),
        )
