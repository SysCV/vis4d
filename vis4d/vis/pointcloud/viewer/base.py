"""Generic classes to visualize and save pointcloud data."""

from __future__ import annotations

import numpy as np

from ..scene import Scene3D


class PointCloudVisualizerBackend:
    """Visualization Backen Interface for Pointclouds."""

    def __init__(
        self,
        class_color_mapping: list[tuple[int, int, int]],
        instance_color_mapping: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Creates a new Open3D visualization backend.

        Args:
            class_color_mapping (list[tuple[int, int ,int]]): List of length
                n_classes that maps each class index to a unique color.
            instance_color_mapping (list[tuple[int, int ,int]], optional): List
                of length n_instances that maps each instance id to a unique
                color. Defaults to None.
        """
        self.scenes: list[Scene3D] = []

        self.class_color_mapping = np.asarray(class_color_mapping)

        if np.any(self.class_color_mapping > 1):  # Color mapping from [0, 255]
            self.class_color_mapping = self.class_color_mapping / 255

        if instance_color_mapping is None:
            self.instance_color_mapping = self.class_color_mapping
        else:
            self.instance_color_mapping = np.asarray(instance_color_mapping)
            if np.any(self.instance_color_mapping > 1):
                self.instance_color_mapping = self.instance_color_mapping / 255

    def create_new_scene(self) -> Scene3D:
        """Creates a new empty scene."""
        self.scenes.append(Scene3D())
        return self.get_current_scene()

    def get_current_scene(self) -> Scene3D:
        """Returns the currently active scene.

        If no scene is available, an new empty one is created.

        Returns:
            Scene3D: current pointcloud scene
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

    def add_scene(self, scene: Scene3D) -> None:
        """Adds a given Scene3D to the visualization.

        Args:
            scene (Scene3D): 3D scene that should be added.
        """
        self.scenes.append(scene)

    def save_to_disk(self, path_to_out_folder: str) -> None:
        """Saves the visualization to disk.

        Args:
            path_to_out_folder (str): Path to output folder
        """
        raise NotImplementedError()
