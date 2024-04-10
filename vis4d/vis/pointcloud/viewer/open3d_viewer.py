"""Open3d visualization backend."""

from __future__ import annotations

import os
from typing import TypedDict

import numpy as np

from vis4d.common.imports import OPEN3D_AVAILABLE
from vis4d.common.typing import NDArrayF64
from vis4d.vis.pointcloud.scene import Scene3D

from .base import PointCloudVisualizerBackend

if OPEN3D_AVAILABLE:
    import open3d as o3d


class PointcloudVisEntry(TypedDict):
    """Entry for a pointcloud to visualize with open3d.

    Only used for typing.
    """

    name: str
    geometry: o3d.geometry.PointCloud


class Open3DVisualizationBackend(PointCloudVisualizerBackend):
    """Backend that uses open3d to visualize potincloud data."""

    def __init__(
        self,
        class_color_mapping: list[tuple[int, int, int]],
        instance_color_mapping: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Creates a new Open3D visualization backend.

        Args:
            color_mapping (NDArrayF64): array of size [n_classes, 3] that maps
                each class index to a unique color.
            class_color_mapping (list[tuple[int, int, int]]): List of length
                n_classes that assigns each class a unique color.
            instance_color_mapping (list[tuple[int, int, int]], optional): List
                of length n_classes that maps each instance id to unqiue color.
                Defaults to None.
        """
        super().__init__(
            class_color_mapping=class_color_mapping,
            instance_color_mapping=instance_color_mapping,
        )

    def save_to_disk(self, path_to_out_folder: str) -> None:
        """Saves the visualization to disk.

        Creates files [colors.ply, classes.ply, instances.ply] for each scene

        Args:
            path_to_out_folder (str): Path to output folder
        """
        for idx, scene in enumerate(self.scenes):
            out_folder = os.path.join(path_to_out_folder, f"scene_{idx:03d}")
            os.makedirs(out_folder, exist_ok=True)

            for vis_pc in self._get_pc_data_for_scene(scene):
                name = vis_pc["name"]
                pc = vis_pc["geometry"]
                o3d.io.write_point_cloud(
                    os.path.join(out_folder, f"{name}.ply"), pc
                )
                print("written", f"{name}.ply")

    def show(self, blocking: bool = False) -> None:
        """Shows the visualization.

        Args:
            blocking (bool): If the visualization should be blocking
                and wait for human input.
        """
        for scene in self.scenes:
            vis_data = []
            vis_data += self._get_pc_data_for_scene(scene)

            o3d.visualization.draw(
                vis_data, non_blocking_and_return_uid=not blocking
            )

    def _get_pc_data_for_scene(
        self, scene: Scene3D
    ) -> list[PointcloudVisEntry]:
        """Converts a given scene to a list of o3d data to visualize.

        Args:
            scene (PointcloudVisEntry): Point cloud scene to visualize
        Returns:
            list[dict[str, Any]]: List of o3d geometries primitives to show.
        """
        xyz, colors, classes, instances = [], [], [], []
        has_classes = False
        has_instances = False

        for pc in scene.points:
            n_pts = pc.xyz.shape[0]

            xyz.append(pc.xyz)
            colors.append(
                pc.colors if pc.colors is not None else np.zeros((n_pts, 3))
            )

            if pc.classes is not None:
                has_classes = True
                col = self.class_color_mapping[
                    pc.classes.squeeze() % self.class_color_mapping.shape[0]
                ]
                classes.append(col)
            else:
                classes.append(np.zeros((n_pts, 3)))

            if pc.instances is not None:
                has_instances = True
                col = self.instance_color_mapping[
                    pc.instances.squeeze()
                    % self.instance_color_mapping.shape[0]
                ]
                instances.append(col)
            else:
                instances.append(np.zeros((n_pts, 3)))

        data: list[PointcloudVisEntry] = []

        data += [
            {
                "name": "colors",
                "geometry": self._create_o3d_cloud(
                    np.concatenate(xyz), np.concatenate(colors)
                ),
            }
        ]
        if has_instances:
            data += [
                {
                    "name": "instances",
                    "geometry": self._create_o3d_cloud(
                        np.concatenate(xyz), np.concatenate(instances)
                    ),
                }
            ]
        if has_classes:
            data += [
                {
                    "name": "classes",
                    "geometry": self._create_o3d_cloud(
                        np.concatenate(xyz), np.concatenate(classes)
                    ),
                }
            ]

        return data

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
