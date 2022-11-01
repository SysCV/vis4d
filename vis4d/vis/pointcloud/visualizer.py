"""Vis4D Visualization tools for analysis and debugging."""
from types import SimpleNamespace
from typing import List

import torch

from vis4d.common.imports import OPEN3D_AVAILABLE
from vis4d.data.const import COMMON_KEYS
from vis4d.data.typing import DictData, ModelOutput
from vis4d.vis.base import Visualizer

if OPEN3D_AVAILABLE:
    from .o3d_backend import Open3DVisualizationBackend


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

    def add_points(self, points: torch.Tensor):
        """Adds a pointcloud to the scene.

        Args:
            points (torch.Tensor): The pointcloud data [n_pts, 3]
        """
        self.points = torch.cat([self.points, _unsqueeze2d(points)])

    def add_semantic_prediction(self, prediction: torch.Tensor):
        """Adds a semantic prediction to the scene.

        Args:
            prediction (torch.Tensor): The semantic prediction shape [n_pts, 1]
        """
        self.semantics.prediction = torch.cat(
            [self.semantics.prediction, _unsqueeze2d(prediction)]
        )

    def add_semantic_groundtruth(self, groundtruth: torch.Tensor):
        """Adds a semantic groundtruth to the scene.

        Args:
            groundtruth (torch.Tensor): The semantic groundtruth
                                        shape [n_pts, 1]
        """
        self.semantics.groundtruth = torch.cat(
            [self.semantics.groundtruth, _unsqueeze2d(groundtruth)]
        )

    def add_colors(self, colors: torch.Tensor):
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
        raise not NotImplementedError()

    def clear(self):
        """Clears all stored data."""
        self.scenes = []


class PointCloudVisualizer(Visualizer):
    """Visualizer that visualizes pointclouds."""

    def __init__(
        self,
        color_mapping: torch.Tensor,
        max_entries=50,
        backend="open3d",
        scene_idx_key: str = "source_index",
        verbose: int = 0,
    ) -> None:
        """Creates a new Pointcloud visualizer.

        Args:
            color_mapping (torch.Tensor): Torch tensor of shape [n_classes, 3]
                                         that assigns each class a unique color
            max_entries (int): Max amount of elements that are visualized
            backend (str): Visualization backend that should be used
            scene_idx_key (str): Name of key that contains the current scene
                                 index. All predictions with the same scene
                                 index will be visualized in the same window
                                 (predictions are conatenated)
            verbose (int): The verbosity of the visualizer
        """
        if backend == "open3d":
            if not OPEN3D_AVAILABLE:
                raise ValueError(
                    f"You have specified the open3d backend."
                    f"But open3d is not installed on this system!"
                )
            else:
                self.visualization_backend = Open3DVisualizationBackend(
                    color_mapping=color_mapping
                )
        else:
            raise ValueError(f"Unknown Point Visualization Backend {backend}")

        self.scene_idx_key = scene_idx_key
        self.verbose = verbose
        self.max_entries = max_entries

        self.clear()

    def _process_non_batched(
        self, model_input: DictData, model_predictions: ModelOutput = {}
    ):
        # Stop visualizing after a predefined amount of data
        if self.num_entries >= self.max_entries:
            return
        self.num_entries += 1

        # Add data to scene
        if self.scene_idx_key not in model_input:
            if self.verbose > 0:  # FIXME, change to log
                print(
                    f"No scene key ('{self.scene_idx_key}') found in current"
                    f"batch! Assuming every prediction is one scene!"
                )
            self.current_scene = self.visualization_backend.create_new_scene()
        else:
            new_scene_idx = model_input[self.scene_idx_key].item()

            if (
                self.current_scene_idx is None
                or self.current_scene_idx != new_scene_idx
            ):
                # Create new scene
                self.current_scene = (
                    self.visualization_backend.create_new_scene()
                )
                self.current_scene_idx = new_scene_idx

        points = model_input[COMMON_KEYS.points3d].detach().cpu()
        if points.shape[-1] != 3:  # Make sure last channel is xyz
            points = points.transpose(-2, -1)
        self.current_scene.add_points(points)

        if COMMON_KEYS.colors3d in model_input:
            colors = model_input[COMMON_KEYS.colors3d].detach().cpu()
            if colors.shape[-1] != 3:
                colors = colors.transpose(-2, -1)
            self.current_scene.add_colors(colors)

        if COMMON_KEYS.semantics3d in model_predictions:
            self.current_scene.add_semantic_prediction(
                model_predictions[COMMON_KEYS.semantics3d]
                .detach()
                .cpu()
                .long()
            )

        if COMMON_KEYS.semantics3d in model_input:
            self.current_scene.add_semantic_groundtruth(
                model_input[COMMON_KEYS.semantics3d].detach().cpu().long()
            )

    def process(
        self, model_input: DictData, model_predictions: ModelOutput = {}
    ):
        """Processes a batch of data and adds it to the visualizer.

        Args:
            model_input (DictData): The input to the model. Must contain
                                    COMMON_KEYS.points3d.
            model_predictions (ModelOutput): The output of the model.
        """
        if COMMON_KEYS.points3d not in model_input:
            raise ValueError(
                f"PointPredVisualizer requires points3d in inputs."
                f"Got: {list(model_input.keys())}"
            )

        pts = model_input[COMMON_KEYS.points3d]

        if len(pts.shape) == 2:  # Data is not batched
            self._process_non_batched(model_input, model_predictions)
        elif len(pts.shape == 3):
            for batch_idx in range(pts.size(0)):
                in_data, pred_data = {}, {}
                for key in model_input:
                    in_data[key] = model_input[key][batch_idx, ...]

                for key in model_predictions:
                    pred_data[key] = model_predictions[key][batch_idx, ...]

                self._process_non_batched(in_data, pred_data)
        else:
            raise ValueError(f"Invalid shape for point data: {pts.shape}")

    def visualize(self):
        """Visualizes to accumulated predictions."""
        self.visualization_backend.visualize()

    def clear(self):
        """Clears all saved data."""
        self.visualization_backend.clear()
        self.current_scene_idx = None
        self.current_scene = None
        self.num_entries = 0

    def save_to_disk(self, path_to_out_folder: str) -> None:
        """Saves the visualization to disk."""
        raise NotImplementedError()  # TODO
