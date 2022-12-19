"""Data structures to store 3D data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import ArrayLike, NDArrayFloat, NDArrayInt


@dataclass
class PointcloudData:
    """Stores pointcloud data for visualization.

    Attributes:
        xyz: Point Coordinates shape [n_pts,3].
        colors: Point Colors shape [n_pts, 3] or None.
        classes: Class ids shape [n_pts] or None.
        instances: Instance ids shape [n_pts] or None.
        num_points: Total number of points.
        num_classes: Total number of classes.
        num_instances: Total number of unique class, instance combinations.
    """

    xyz: NDArrayFloat
    colors: NDArrayInt
    classes: NDArrayInt
    instances: NDArrayInt

    num_points: int
    num_classes: int
    num_instances: int

    def __init__(
        self,
        xyz: ArrayLike,
        colors: ArrayLike | None = None,
        classes: ArrayLike | None = None,
        instances: ArrayLike | None = None,
    ) -> None:
        """Creates a new pointcloud.

        Args:
            xyz (ArrayLike): Coordinates for each point shape [n_pts, 3]
            colors (ArrayLike | None, optional): Colors for each point encoded
                as rgb [n_pts, 3] in the range (0,255). Defaults to None.
            classes (ArrayLike | None, optional): Class id for each point
                shape [n_pts]. Defaults to None.
            instances (ArrayLike | None, optional): Instance id for each point.
                shape [n_pts]. Defaults to None.
        """
        self.xyz = array_to_numpy(xyz, n_dims=2)
        self.colors = array_to_numpy(colors, n_dims=2)
        self.classes = array_to_numpy(classes, n_dims=1)
        self.instances = array_to_numpy(instances, n_dims=1)

        # Assing other properties. Number points, ...
        self.num_points = self.xyz.shape[0]

        if self.classes is not None:
            self.num_classes = len(np.unique(self.classes))

        if self.instances is not None:
            if self.classes is None:
                self.num_instances = len(np.unique(self.instances))
            else:
                self.num_instances = len(
                    np.unique(
                        self.classes * np.max(self.instances) + self.instances
                    )
                )


class Scene3D:
    """Stores the data for a 3D scene.

    This Scene3D object can be used to be visualized by any 3D viewer.

    Attributes:
        pointclouds (list[PointcloudData]): Stores all pointclouds that
            have been registered for this scene so far.
        pointclouds (list[NDArrayFloat]): Stores a transformation matrix
            (SE3, shape (4,4)) for each pointcloud.
    """

    def __init__(self) -> None:
        """Creates a new, empty scene."""
        self.pointclouds: list[PointcloudData] = []
        self.transforms: list[NDArrayFloat] = []

    def add_pointcloud(
        self,
        xyz: ArrayLike,
        colors: ArrayLike | None = None,
        classes: ArrayLike | None = None,
        instances: ArrayLike | None = None,
        transform: ArrayLike | None = None,
    ) -> Scene3D:
        """Adds a pointcloud to the 3D Scene.

        Args:
            xyz (ArrayLike): Coordinates for each point shape [n_pts, 3] in the
                current local frame.
            colors (ArrayLike | None, optional): Colors for each point encoded
                as rgb [n_pts, 3] in the range (0,255) or (0,1).
                Defaults to None.
            classes (ArrayLike | None, optional): Class id for each point
                shape [n_pts]. Defaults to None.
            instances (ArrayLike | None, optional): Instance id for each point.
                shape [n_pts]. Defaults to None.
            transform (ArrayLike | None, optional): Transformation matrix
                shape [4,4] that transforms points from the current local frame
                to a fixed global frame. Defaults to None which is the identity
                matrix.

        Returns:
            Scene3D: Returns 'self' to chain calls.
        """
        se3_tf = (
            array_to_numpy(transform, n_dims=2)
            if transform is not None
            else np.eye(4)
        )

        assert se3_tf.shape == (
            4,
            4,
        ), "Shape of the provided transform not valid."
        self.pointclouds.append(
            PointcloudData(xyz, colors, classes, instances)
        )
        self.transforms.append(se3_tf)
        return self
