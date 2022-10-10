""""""
import math
import unittest

import torch

from .orthogonal_transform_loss import OrthogonalTransformRegularizationLoss


class TestOrthogonalTransformLoss(unittest.TestCase):
    """Tests the losses needed for pointnet segmentation"""

    def test_regularization_loss(self) -> None:

        bs = 4
        loss = OrthogonalTransformRegularizationLoss()
        transforms = [torch.eye(s).repeat(bs, 1, 1) for s in [3, 64]]
        # Loss should be zero for identity matrix
        self.assertAlmostEqual(loss(transforms).item(), 0)

        transforms = [
            torch.tensor(
                [
                    math.cos(a),
                    math.sin(a),
                    0,
                    -math.sin(a),
                    math.cos(a),
                    0,
                    0,
                    0,
                    1,
                ]
            ).reshape(1, 3, 3)
            for a in [0.1, 0.2, 0.3]
        ]
        # Loss should be zero for rotation matrix
        self.assertAlmostEqual(loss(transforms).item(), 0, places=4)

        loss = OrthogonalTransformRegularizationLoss()
        transforms = [torch.eye(s).repeat(bs, 1, 1) + 0.1 for s in [3]]
        self.assertNotEqual(loss(transforms).item(), 0)

        # Check numerical
        input_tensor = torch.Tensor(
            [
                [1.0316, 0.0100, 0.0148],
                [0.0316, 1.0100, 0.0148],
                [0.0316, 0.0100, 1.0148],
            ]
        )
        self.assertAlmostEqual(
            loss([input_tensor.unsqueeze(0)]).item(), 0.12318, places=4
        )
