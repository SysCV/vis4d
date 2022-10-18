"""Open3d visualization backend"""
from typing import List, Union

import numpy as np
import open3d as o3d
import torch

from .base import PointcloudScene, PointCloudVisualizerBackend


class Open3DVisualizationBackend(PointCloudVisualizerBackend):
    """Backend that uses open3d to visualize potincloud data"""

    def __init__(
        self, color_mapping: torch.Tensor, use_same_window=True
    ) -> None:
        """Creates a new Open3D visualization backend.

        Args:
            color_mapping (tensor): Tensor of size [n_classes, 3] that maps
            each class index to a unique color.
            use_same_window (bool): If true, visualizes all predictions in
                                    same window
        """
        super().__init__(color_mapping=color_mapping)

        self.use_same_window = use_same_window

    def visualize(self):
        """Visualizes the collected data."""
        for scene in self.scenes:
            data_to_visualize = self._get_vis_data_for_scene(scene)
            if self.use_same_window:
                # Draw everything in same windows
                o3d.visualization.draw_geometries(data_to_visualize)
            else:
                # Use separate windows for each visualization (colors,
                # semantics ,....)
                for i in range(len(data_to_visualize // 2)):
                    o3d.visualization.draw_geometries(
                        data_to_visualize[(2 * i) : (2 * i + 1)]
                    )

    @staticmethod
    def _create_o3d_cloud(
        points: np.array, colors: np.array
    ) -> o3d.geometry.PointCloud:
        """Creates a o3d pointcloud from poitns and colors."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _get_vis_data_for_scene(
        self, scene: PointcloudScene
    ) -> List[Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]]:
        """Converts a given scene to a list of o3d data to visualize."""

        # Visualize each scene
        points = scene.points.numpy()
        colors = scene.colors.numpy()
        sem_pred = scene.semantics.prediction.numpy()
        sem_gt = scene.semantics.groundtruth.numpy()

        pts_bounds = np.max(np.abs(points), axis=0)  # max bounds xyz

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]
        )

        data_to_visualize = [origin]

        # Visualize pc
        data_to_visualize.append(self._create_o3d_cloud(points, colors))

        if len(sem_pred > 0):
            vis_idx = (
                len(data_to_visualize) // 2 if self.use_same_window else 0
            )
            offset = vis_idx * pts_bounds
            offset[2] = 0  # Only move in x,y

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.6, origin=[*offset]
            )
            data_to_visualize.append(origin)
            colors = self.color_mapping[
                scene.semantics.prediction.squeeze()
            ].numpy()
            data_to_visualize.append(
                self._create_o3d_cloud(points + offset, colors)
            )

        if len(sem_gt > 0):
            vis_idx = (
                len(data_to_visualize) // 2 if self.use_same_window else 0
            )
            offset = vis_idx * pts_bounds
            offset[2] = 0  # Only move in x,y

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.6, origin=[*offset]
            )
            data_to_visualize.append(origin)
            colors = self.color_mapping[
                scene.semantics.groundtruth.squeeze()
            ].numpy()
            data_to_visualize.append(
                self._create_o3d_cloud(points + offset, colors)
            )

        return data_to_visualize
