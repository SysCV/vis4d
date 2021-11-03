"""Projection tests."""
import unittest

import torch

from vis4d.struct import Intrinsics

from .projection import project_points, unproject_points


class TestProjection(unittest.TestCase):
    """Testcases for projection functions."""

    points_3d = torch.tensor(
        [[0, 0, 10], [1, 1, 5], [-2, -2, 20]], dtype=torch.float32
    )
    intrinsic_matrix = torch.eye(3, dtype=torch.float32)
    intrinsic_matrix[0, 0] = 700
    intrinsic_matrix[1, 1] = 700
    # e.g. resolution 1920x1280
    intrinsic_matrix[0, 2] = 1920 / 2
    intrinsic_matrix[1, 2] = 1280 / 2
    intrinsics = Intrinsics(intrinsic_matrix)

    points_2d = torch.tensor(
        [[1920 / 2, 1280 / 2], [1100, 780], [890, 570]], dtype=torch.float32
    )

    def test_project_points(self) -> None:
        """Test project_points function."""
        proj_points = project_points(self.points_3d, self.intrinsics)
        self.assertTrue(torch.isclose(proj_points, self.points_2d).all())
        proj_points = project_points(
            self.points_3d.unsqueeze(0), self.intrinsics
        )
        self.assertEqual(tuple(proj_points.shape), (1, 3, 2))
        self.assertTrue(torch.isclose(proj_points, self.points_2d).all())

    def test_unproject_points(self) -> None:
        """Test unproject_points function."""
        unproj_points = unproject_points(
            self.points_2d, self.points_3d[:, -1], self.intrinsics
        )
        self.assertTrue(torch.isclose(unproj_points, self.points_3d).all())
        unproj_points = unproject_points(
            self.points_2d.unsqueeze(0),
            self.points_3d[:, -1].unsqueeze(0),
            self.intrinsics,
        )
        self.assertEqual(tuple(unproj_points.shape), (1, 3, 3))
        self.assertTrue(torch.isclose(unproj_points, self.points_3d).all())
