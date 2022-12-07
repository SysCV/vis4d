"""Transform tests."""
import math
import unittest

import torch

from vis4d.op.geometry.rotation import euler_angles_to_matrix
from vis4d.op.geometry.transform import (
    inverse_pinhole,
    inverse_rigid_transform,
    transform_points,
)


class TestTransform(unittest.TestCase):
    """Testcases for transform functions."""

    points = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=torch.float32
    )
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = euler_angles_to_matrix(
        torch.Tensor([[math.pi / 2, 0, 0]])
    )

    points_target = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [1, -1, 0], [1, 0, 1]], dtype=torch.float32
    )

    def test_transform_points(self) -> None:
        """Test transform_points function."""
        # (B, N, D) / (B, D+1, D+1)
        points_transformed = transform_points(
            self.points.unsqueeze(0), self.transform.unsqueeze(0)
        )
        self.assertTrue(
            torch.isclose(
                points_transformed, self.points_target, atol=1e-5
            ).all()
        )

        # (N, D) / (D+1, D+1)
        points_transformed = transform_points(self.points, self.transform)
        self.assertTrue(
            torch.isclose(
                points_transformed, self.points_target, atol=1e-5
            ).all()
        )

        # (N, D) / (B, D+1, D+1)
        points_transformed = transform_points(
            self.points, self.transform.unsqueeze(0)
        )
        self.assertTrue(
            torch.isclose(
                points_transformed, self.points_target, atol=1e-5
            ).all()
        )

        # (B, N, D) / (D+1, D+1)
        points_transformed = transform_points(
            self.points.unsqueeze(0), self.transform
        )
        self.assertTrue(
            torch.isclose(
                points_transformed, self.points_target, atol=1e-5
            ).all()
        )


def test_inverse_rigid_transform():
    """Testcase for inverse rigid transform function."""
    angles = torch.tensor([math.pi, math.pi / 2, math.pi / 4])
    transform = torch.eye(4)
    transform[:3, :3] = euler_angles_to_matrix(angles)
    transform[:3, 3] = torch.tensor([0.5, -0.5, 1.0])
    inverse_t = inverse_rigid_transform(transform)
    assert torch.isclose(inverse_t, transform.inverse()).all()
    transform = transform.unsqueeze(0)
    inverse_t = inverse_rigid_transform(transform)
    assert torch.isclose(inverse_t, transform.inverse()).all()


def test_inverse_pinhole():
    """Testcase for inverse pinhole function."""
    intrinsic_matrix = torch.eye(3, dtype=torch.float32)
    intrinsic_matrix[0, 0] = 700
    intrinsic_matrix[1, 1] = 700
    # e.g. resolution 1920x1280
    intrinsic_matrix[0, 2] = 1920 / 2
    intrinsic_matrix[1, 2] = 1280 / 2

    inverse_i = inverse_pinhole(intrinsic_matrix)
    assert torch.isclose(inverse_i, intrinsic_matrix.inverse()).all()

    intrinsic_matrix = intrinsic_matrix.unsqueeze(0)
    inverse_i = inverse_pinhole(intrinsic_matrix)
    assert torch.isclose(inverse_i, intrinsic_matrix.inverse()).all()
