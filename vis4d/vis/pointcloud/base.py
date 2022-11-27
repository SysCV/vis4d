"""Generic classes to save pointcloud data."""
from __future__ import annotations

from typing import overload

import numpy as np

from vis4d.common.typing import NDArrayF64, NDArrayI64, NDArrayNumber


@overload
def _unsqueeze2d(array: NDArrayF64) -> NDArrayF64:
    ...


@overload
def _unsqueeze2d(array: NDArrayI64) -> NDArrayI64:
    ...


def _unsqueeze2d(array: NDArrayNumber) -> NDArrayNumber:
    """Adds an empty dimension from the right if target is not 2D.

    Args:
        array: Numpy array to unsqueeze

    Returns:
        Unsqueezed 2D array
    """
    return array[..., None] if len(array.shape) == 1 else array


class PointcloudScene:
    """Stores the data for a 3D scene. (points, semantics, ...)."""

    def __init__(self) -> None:
        """Creates a new, empty scene."""
        self.points = np.zeros((0, 3))  # xyz
        self.colors = np.zeros((0, 3))  # rgb
        self.semantics: NDArrayI64 = np.zeros((0, 1), dtype=np.int64)
        self.instances: NDArrayI64 = np.zeros((0, 1), dtype=np.int64)

    def add_points(self, points: NDArrayF64) -> None:
        """Adds a pointcloud to the scene.

        Args:
            points (NDArrayF64): The pointcloud data [n_pts, 3]
        """
        self.points = np.concatenate([self.points, _unsqueeze2d(points)])

    def add_semantics(self, semantics: NDArrayI64) -> None:
        """Adds semantic information to the scene.

        Args:
            semantics (NDArrayI64): The semantic prediction shape [n_pts, 1]
        """
        self.semantics = np.concatenate(
            [self.semantics, _unsqueeze2d(semantics)]
        )

    def add_instances(self, instances: NDArrayI64) -> None:
        """Adds semantic information to the scene.

        Args:
            instances (NDArrayI64): The semantic prediction shape [n_pts, 1]
        """
        self.instances = np.concatenate(
            [self.instances, _unsqueeze2d(instances)]
        )

    def add_colors(self, colors: NDArrayF64) -> None:
        """Adds color information tot he scene.

        Args:
            colors (NDArrayF64): The color data [n_pts, 3] ranging from [0,1].
        """
        self.colors = np.concatenate([self.colors, _unsqueeze2d(colors)])


class PointCloudVisualizerBackend:
    """Visualization Backen Interface for Pointclouds."""

    def __init__(
        self,
        class_color_mapping: list[tuple[float, float, float]],
        instance_color_mapping: list[tuple[float, float, float]] | None = None,
    ) -> None:
        """Creates a new Open3D visualization backend.

        Args:
            class_color_mapping (array): Array of size [n_classes, 3] that maps
                each class index to a unique color.
            instance_color_mapping (array): Array of size [n_instances, 3] that
                maps each instance id to a unique color.
        """
        self.scenes: list[PointcloudScene] = []

        self.class_color_mapping = np.asarray(class_color_mapping)

        if np.any(self.class_color_mapping > 1):  # Color mapping from [0, 255]
            self.class_color_mapping = self.class_color_mapping / 255

        if instance_color_mapping is None:
            self.instance_color_mapping = self.class_color_mapping
        else:
            self.instance_color_mapping = np.asarray(instance_color_mapping)
            if np.any(self.instance_color_mapping > 1):
                self.instance_color_mapping = self.instance_color_mapping / 255

    def create_new_scene(self) -> PointcloudScene:
        """Creates a new empty scene."""
        self.scenes.append(PointcloudScene())
        return self.get_current_scene()

    def get_current_scene(self) -> PointcloudScene:
        """Returns the currently active scene.

        If no scene is available, an new empty one is created.

        Returns:
            PointcloudScene: current pointcloud scene
        """
        if (len(self.scenes)) == 0:
            return self.create_new_scene()

        return self.scenes[-1]

    def show(self, blocking: bool = True) -> None:
        """Shows the visualization.

        Args:
            blocking (bool): If the visualization should be blocking
                             and wait for human input
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Clears all stored data."""
        self.scenes = []

    def save_to_disk(self, path_to_out_folder: str) -> None:
        """Saves the visualization to disk.

        Args:
            path_to_out_folder (str): Path to output folder
        """
        raise NotImplementedError()
