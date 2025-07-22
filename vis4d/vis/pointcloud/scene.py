"""Data structures to store 3D data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import ArrayLike, NDArrayFloat, NDArrayInt


@dataclass
class BoundingBoxData:
    """Stores bounding box data for visualization.

    Attributes:
        corners (NDArrayFloat): Corners of the bounding box shape [8, 3].
        color (NDArrayFloat): Colors of the bounding box shape [3].
        class (int | None): Class id of the bounding box. Defaults to None.
        instance (int | None): Instance id of the bounding box.
            Defaults to None.
        score (float | None): Score of the bounding box. Defaults to None.
    """

    corners: NDArrayFloat
    color: NDArrayFloat | None
    class_: int | None
    instance: int | None
    score: float | None

    def transform(self, transform: NDArrayFloat) -> BoundingBoxData:
        """Transforms the bounding box.

        Args:
            transform (NDArrayFloat): Transformation matrix shape [4,4] that
                transforms points from the current local frame to a fixed
                global frame.

        Returns:
            BoundingBoxData: Returns a new bounding box with the transformed
                points.
        """
        assert transform.shape == (
            4,
            4,
        ), "Shape of the provided transform not valid."
        return BoundingBoxData(
            (transform[:3, :3] @ self.corners.T).T + transform[:3, -1],
            self.color,
            self.class_,
            self.instance,
            self.score,
        )


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
    colors: NDArrayFloat | None
    classes: NDArrayInt | None
    instances: NDArrayInt | None

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
        self.xyz = array_to_numpy(xyz, n_dims=2, dtype=np.float32)
        self.colors = array_to_numpy(colors, n_dims=2, dtype=np.float32)
        self.classes = array_to_numpy(classes, n_dims=1, dtype=np.int32)
        self.instances = array_to_numpy(instances, n_dims=1, dtype=np.int32)

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

    def transform(self, transform: NDArrayFloat) -> PointcloudData:
        """Transforms the pointcloud.

        Args:
            transform (NDArrayFloat): Transformation matrix shape [4,4] that
                transforms points from the current local frame to a fixed
                global frame.

        Returns:
            PointcloudData: Returns a new pointcloud with the transformed
                points.
        """
        assert transform.shape == (
            4,
            4,
        ), "Shape of the provided transform not valid."
        return PointcloudData(
            (transform[:3, :3] @ self.xyz.T).T + transform[:3, -1],
            self.colors,
            self.classes,
            self.instances,
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
        self._pointclouds: list[tuple[PointcloudData, NDArrayFloat]] = []
        self._bounding_boxes: list[tuple[BoundingBoxData, NDArrayFloat]] = []

    @staticmethod
    def _parse_se3_transform(transform: ArrayLike | None) -> NDArrayFloat:
        """Parses a SE3 transformation matrix.

        Args:
            transform (ArrayLike | None): Transformation matrix shape [4,4]
                that transforms points from the current local frame to a fixed
                global frame.

        Returns:
            NDArrayFloat: Returns a valid SE3 transformation matrix.
        """
        tf = array_to_numpy(transform, n_dims=2, dtype=np.float32)

        if tf is None:
            return np.eye(4)

        assert tf.shape == (
            4,
            4,
        ), "Shape of the provided transform not valid."
        return tf

    def add_bounding_box(
        self,
        corners: ArrayLike,
        color: ArrayLike | None,
        class_: int | None,
        instance: int | None,
        score: float | None,
        transform: ArrayLike | None = None,
    ) -> Scene3D:
        """Adds a bounding box to the 3D Scene.

        Args:
            corners (ArrayLike): Corners of the bounding box shape [8, 3].
            color (ArrayLike | None): Color of the bounding box shape [3].
            class_ (int | None): Class id of the bounding box.
                Defaults to None.
            instance (int | None): Instance id of the bounding box.
                Defaults to None.
            score (float | None): Score of the bounding box. Defaults to None.
            transform (ArrayLike | None): Transformation matrix shape [4,4]
                that transforms points from the current local frame to a fixed
                global frame.

        Returns:
            Scene3D: Returns 'self' to chain calls.
        """
        corners_np = array_to_numpy(corners, n_dims=2, dtype=np.float32)
        colors_np = array_to_numpy(color, n_dims=1, dtype=np.float32)
        self._bounding_boxes.append(
            (
                BoundingBoxData(
                    corners_np,
                    colors_np,
                    class_,
                    instance,
                    score,
                ),
                self._parse_se3_transform(transform),
            ),
        )
        return self

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
        self._pointclouds.append(
            (
                PointcloudData(xyz, colors, classes, instances),
                self._parse_se3_transform(transform),
            )
        )
        return self

    @property
    def bounding_boxes(self) -> list[BoundingBoxData]:
        """Returns all bounding boxes in the scene.

        Returns:
            list[BoundingBoxData]: List of all bounding boxes in the scene.
        """
        return [bbox.transform(tf) for (bbox, tf) in self._bounding_boxes]

    @property
    def points(self) -> list[PointcloudData]:
        """Returns all points of all pointclouds in the scene.

        Returns:
            List[PointcloudData]: Data information for all points in the scene.
                Providing information about the points, colors, classes and
                instances.
        """
        return [pc.transform(tf) for (pc, tf) in self._pointclouds]
