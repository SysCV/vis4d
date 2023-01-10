"""Open3d visualization backend."""
from __future__ import annotations

import os

import numpy as np
import open3d as o3d

from vis4d.common.typing import NDArrayF64
from vis4d.vis.pointcloud.base import PointCloudVisualizerBackend
from vis4d.vis.pointcloud.scene import Scene3D


class Open3DVisualizationBackend(PointCloudVisualizerBackend):
    """Backend that uses open3d to visualize potincloud data."""

    def __init__(
        self,
        class_color_mapping: list[tuple[float, float, float]],
        instance_color_mapping: list[tuple[float, float, float]] | None = None,
        use_same_window: bool = True,
    ) -> None:
        """Creates a new Open3D visualization backend.

        Args:
            color_mapping (NDArrayF64): array of size [n_classes, 3] that maps
                each class index to a unique color.
            class_color_mapping (list[tuple[float, float, float]]): List of
                size [n_classes, 3] that maps each class id to a unique color.
            instance_color_mapping (list[tuple[float, float, float]]): List
                of size [n_classes, 3] that maps each instance id to unqiue
                color.
            use_same_window (bool): If true, visualizes all predictions in
                same window by creating mutiple pointclouds which are offset
                by each other. If false creates a window for each attribute.
        """
        super().__init__(
            class_color_mapping=class_color_mapping,
            instance_color_mapping=instance_color_mapping,
        )
        self.use_same_window = use_same_window

    def save_to_disk(self, path_to_out_folder: str) -> None:
        """Saves the visualization to disk.

        Creates files [colors.ply, classes.ply, instances.ply] for each scene

        Args:
            path_to_out_folder (str): Path to output folder
        """
        for idx, scene in enumerate(self.scenes):
            out_folder = os.path.join(path_to_out_folder, f"scene_{idx:03d}")
            os.makedirs(out_folder, exist_ok=True)
            colors = self._create_o3d_cloud(scene.points, scene.colors)
            o3d.io.write_point_cloud(
                os.path.join(out_folder, "colors.ply"), colors
            )

            if len(scene.classes) > 0:
                colors = self.class_color_mapping[
                    scene.classes.squeeze() % self.class_color_mapping.shape[0]
                ]
                classes = self._create_o3d_cloud(scene.points, colors)
                o3d.io.write_point_cloud(
                    os.path.join(out_folder, "classes.ply"), classes
                )

            if len(scene.instances) > 0:
                colors = self.instance_color_mapping[
                    scene.instances.squeeze()
                    % self.instance_color_mapping.shape[0]
                ]
                instances = self._create_o3d_cloud(scene.points, colors)
                o3d.io.write_point_cloud(
                    os.path.join(out_folder, "instances.ply"), instances
                )

    def show(self, blocking: bool = True) -> None:
        """Shows the visualization.

        Args:
            blocking (bool): If the visualization should be blocking
                             and wait for human input
        """
        for scene in self.scenes:
            data_to_visualize = self._get_vis_data_for_scene(scene)
            if self.use_same_window:
                # Draw everything in the same window
                data = []
                for d in data_to_visualize:
                    data += d
                if blocking:
                    o3d.visualization.draw_geometries(  # pylint: disable=no-member,line-too-long
                        data
                    )
                else:
                    v = (
                        o3d.visualization.Visualizer()  # pylint: disable=no-member,line-too-long
                    )
                    v.create_window()
                    for geom in data:
                        v.add_geometry(geom)
            else:
                # Use separate windows for each visualization (colors,
                # semantics ,....)
                for data in data_to_visualize:
                    if blocking:
                        o3d.visualization.draw_geometries(  # pylint: disable=no-member,line-too-long
                            data
                        )
                    else:
                        v = (
                            o3d.visualization.Visualizer()  # pylint: disable=no-member,line-too-long
                        )
                        v.create_window()
                        for geom in data:
                            v.add_geometry(geom)

    @staticmethod
    def _create_o3d_cloud(
        points: NDArrayF64,
        colors: NDArrayF64 | None = None,
        normals: NDArrayF64 | None = None,
    ) -> o3d.geometry.PointCloud:
        """Creates a o3d pointcloud from poitns and colors.

        Args:
            points (NDArrayF64): xyz coordinates of the points
            colors (NDArrayF64, optional): Colors of the points
            normals (NDArrayF64, optional): Surface normals

        Returns:
            o3d.geometry.PointCloud: o3d pointcloud with the given attributes
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None and len(normals) > 0:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        return pcd

    def _get_vis_data_for_scene(
        self, scene: Scene3D
    ) -> list[list[o3d.geometry.TriangleMesh | o3d.geometry.PointCloud]]:
        """Converts a given scene to a list of o3d data to visualize.

        Args:
            scene (PointCloudScene): Point cloud scene to visualize
        Returns:
            list[list[o3d.geometry]]: List of o3d geometries to show.
        """
        # Visualize each scene
        points = np.zeros((0, 3))
        colors = np.zeros((0, 3))
        semantics = np.zeros(0).astype(int)
        instances = np.zeros(0).astype(int)

        for pc in scene.pointclouds:
            points = np.concatenate([points, pc.xyz])
            # FIXME. This will only work if ALL points have colors, instances
            # etc or No points have, Otherwise we will have missmatched
            # assignements
            if pc.colors is not None:
                colors = np.concatenate([colors, pc.colors])
            if pc.classes is not None:
                semantics = np.concatenate([semantics, pc.classes])
            if pc.instances is not None:
                instances = np.concatenate([instances, pc.instances])

        pts_bounds = np.max(np.abs(points), axis=0) * 2  # max bounds xyz

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]
        )
        data_to_visualize = []

        if len(colors) > 0:
            data_to_visualize += [
                [origin, self._create_o3d_cloud(points, colors)]
            ]

        if len(semantics) > 0:
            # Move origin for visualization
            # if we are plotting in the same window
            vis_idx = len(data_to_visualize) if self.use_same_window else 0
            offset = vis_idx * pts_bounds
            offset[2] = 0  # Only move in x,y

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.6, origin=[*offset]
            )
            colors = self.class_color_mapping[
                semantics.squeeze() % self.class_color_mapping.shape[0]
            ]
            data_to_visualize.append(
                [origin, self._create_o3d_cloud(points + offset, colors)]
            )

        if len(instances) > 0:
            # Move origin for visualization
            # if we are plooting in the same window
            vis_idx = len(data_to_visualize) if self.use_same_window else 0
            offset = vis_idx * pts_bounds
            offset[2] = 0  # Only move in x,y

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.6, origin=[*offset]
            )
            colors = self.instance_color_mapping[
                instances.squeeze() % self.instance_color_mapping.shape[0]
            ]
            data_to_visualize.append(
                [origin, self._create_o3d_cloud(points + offset, colors)]
            )

        return data_to_visualize
