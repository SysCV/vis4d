"""Transform tests."""
import math
import unittest

import torch

from .rotation import euler_angles_to_matrix
from .transform import transform_points


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
