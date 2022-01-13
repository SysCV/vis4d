"""Projection tests."""
import unittest

import torch

from vis4d.struct import Intrinsics

from .projection import (
    generate_depth_map,
    generate_projected_point_mask,
    project_points,
    unproject_points,
)


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

    image_width = 1920
    image_height = 1280

    depth_map = torch.zeros((image_height, image_width))
    depth_map[
        points_2d[:, 1].type(torch.long), points_2d[:, 0].type(torch.long)
    ] = points_3d[:, 2]

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

    def test_generate_projected_point_mask(self) -> None:
        """Test generate projected point mask."""
        mask = generate_projected_point_mask(
            self.points_3d[:, 2],
            self.points_2d,
            self.image_width,
            self.image_height,
        )
        self.assertTrue(
            torch.isclose(mask, torch.ones(3, dtype=torch.bool)).all()
        )

    def test_generate_depth_map(self) -> None:
        """Test generate depth map function."""
        depth_map, pts2d, depths, _ = generate_depth_map(
            self.points_3d,
            self.intrinsics,
            self.image_width,
            self.image_height,
            torch.ones(3, dtype=torch.bool),
        )
        self.assertTrue(torch.isclose(pts2d, self.points_2d).all())
        self.assertTrue(
            torch.isclose(depths, self.points_3d[:, -1].unsqueeze(0)).all()
        )
        self.assertTrue(torch.isclose(depth_map, self.depth_map).all())
