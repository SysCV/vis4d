"""Generic classes to save pointcloud data."""
from types import SimpleNamespace
from typing import List

import torch


def _unsqueeze2d(arg: torch.Tensor) -> torch.Tensor:
    """Adds an empty dimension from the right if target is not 2D."""
    if len(arg.shape) == 1:
        return arg.unsqueeze(-1)
    return arg


class PointcloudScene:
    """Stores the data for a 3D scene. (points, semantics, ...)."""

    def __init__(self) -> None:
        """Creates a new, empty scene."""
        self.points = torch.zeros(0, 3)  # xyz
        self.colors = torch.zeros(0, 3)  # rgb
        self.semantics = SimpleNamespace(
            prediction=torch.zeros(0, 1).long(),
            groundtruth=torch.zeros(0, 1).long(),
        )

    def add_points(self, points: torch.Tensor) -> None:
        """Adds a pointcloud to the scene.

        Args:
            points (torch.Tensor): The pointcloud data [n_pts, 3]
        """
        self.points = torch.cat([self.points, _unsqueeze2d(points)])

    def add_semantic_prediction(self, prediction: torch.Tensor) -> None:
        """Adds a semantic prediction to the scene.

        Args:
            prediction (torch.Tensor): The semantic prediction shape [n_pts, 1]
        """
        self.semantics.prediction = torch.cat(
            [self.semantics.prediction, _unsqueeze2d(prediction)]
        )

    def add_semantic_groundtruth(self, groundtruth: torch.Tensor) -> None:
        """Adds a semantic groundtruth to the scene.

        Args:
            groundtruth (torch.Tensor): The semantic groundtruth
                                        shape [n_pts, 1]
        """
        self.semantics.groundtruth = torch.cat(
            [self.semantics.groundtruth, _unsqueeze2d(groundtruth)]
        )

    def add_colors(self, colors: torch.Tensor) -> None:
        """Adds color information tot he scene.

        Args:
            colors (torch.Tensor): The color data [n_pts, 3]
        """

        self.colors = torch.cat([self.colors, _unsqueeze2d(colors)])


class PointCloudVisualizerBackend:
    """Visualization Backen Interface for Pointclouds."""

    def __init__(self, color_mapping: torch.Tensor) -> None:
        """Creates a new Open3D visualization backend.

        Args:
            color_mapping (tensor): Tensor of size [n_classes, 3] that maps
            each class index to a unique color.
        """
        self.scenes: List[PointcloudScene] = []

        if (color_mapping > 1).any():  # Color mapping from [0, 255]
            self.color_mapping = color_mapping / 255
        else:
            self.color_mapping = color_mapping

    def create_new_scene(self) -> PointcloudScene:
        """Creates a new empty scene."""
        self.scenes.append(PointcloudScene())
        return self.get_current_scene()

    def get_current_scene(self) -> PointcloudScene:
        """Returns the currently active scene."""
        if (len(self.scenes)) == 0:
            return self.create_new_scene()

        return self.scenes[-1]

    def visualize(self):
        """Visualizes the stored data."""
        raise NotImplementedError

    def clear(self):
        """Clears all stored data."""
        self.scenes = []
