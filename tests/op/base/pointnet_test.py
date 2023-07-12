"""Testcases for pointnet."""
import unittest

import torch

from vis4d.op.base.pointnet import (
    LinearTransform,
    PointNetEncoder,
    PointNetSegmentation,
)


def convert_batched_to_sparse(
    batched: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a batched tensor to a sparse tensor.

    Args:
        batched: (tensor) shape [B, C, N]

    Returns:
        sparse tensor values: (tensor) shape [C, N*B]
        indices: (tensor) shape [N*B]
    """
    batch_size, _, n_pts = batched.shape
    return torch.cat(list(batched), dim=-1), torch.arange(
        batch_size
    ).repeat_interleave(n_pts)


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

            sparse_in, sparse_idx = convert_batched_to_sparse(pts)
            out_sparse = layer(sparse_in, sparse_idx)
            self.assertTrue(torch.allclose(out, out_sparse))

            # # Permute idxs
            # perm_idx = torch.randperm(len(sparse_idx))
            # sparse_in = sparse_in[:, perm_idx]
            # sparse_idx = sparse_idx[perm_idx]
            # out_sparse = layer(sparse_in, sparse_idx)
            # self.assertTrue(torch.allclose(out[..., perm_idx], out_sparse))

    #         # pts_splitted =

    def test_encoder_shape(self) -> None:
        """Tests the shapes of the full pointnet encoder."""
        for n_features in (3, 6, 9):
            for out_dim in (1024, 2048):
                mlp_dimensions = [[64, 64], [64, 128]]
                pts = torch.rand(self.batch_size_, n_features, self.n_pts_)
                encoder = PointNetEncoder(
                    in_dimensions=n_features, out_dimensions=out_dim
                )
                encoder.eval()
                out = encoder(pts)

                self.assertEqual(out.features.shape[0], self.batch_size_)
                self.assertEqual(out.features.shape[1], out_dim)

                self.assertEqual(len(out.transformations), len(mlp_dimensions))
                sparse_in, sparse_idx = convert_batched_to_sparse(pts)
                out_sparse = encoder(sparse_in, sparse_idx)

                self.assertTrue(
                    torch.allclose(out.features, out_sparse.features)
                )

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

        # Check sparse prediction matches dense prediction
        sparse_in, sparse_idx = convert_batched_to_sparse(pts)
        out_sparse = segmenter(sparse_in, sparse_idx)
        dense_pred, _ = convert_batched_to_sparse(predict_logits)

        self.assertTrue(
            torch.allclose(dense_pred, out_sparse.class_logits.squeeze(0))
        )
