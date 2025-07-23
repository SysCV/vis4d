"""Vis4D Visualization tools for analysis and debugging."""

from __future__ import annotations

from vis4d.common.imports import OPEN3D_AVAILABLE
from vis4d.common.typing import ArgsType, NDArrayF64, NDArrayI64
from vis4d.vis.base import Visualizer
from vis4d.vis.pointcloud.scene import Scene3D
from vis4d.vis.pointcloud.viewer import PointCloudVisualizerBackend
from vis4d.vis.util import DEFAULT_COLOR_MAPPING

if OPEN3D_AVAILABLE:
    from .viewer.open3d_viewer import Open3DVisualizationBackend


# TODO: Check typing
class PointCloudVisualizer(Visualizer):
    """Visualizer that visualizes pointclouds."""

    def __init__(
        self,
        *args: ArgsType,
        backend: str = "open3d",
        class_color_mapping: list[
            tuple[int, int, int]
        ] = DEFAULT_COLOR_MAPPING,
        instance_color_mapping: list[
            tuple[int, int, int]
        ] = DEFAULT_COLOR_MAPPING,
        **kwargs: ArgsType,
    ) -> None:
        """Creates a new Pointcloud visualizer.

        Args:
            backend (str): Visualization backend that should be used. Choice
                of [open3d].
            class_color_mapping (list[tuple[int, int, int]], optional): List
                of length n_classes that assigns each class a unique color.
            instance_color_mapping (list[tuple[int, int, int]], optional): List
                of length n_classes that assigns each class a unique color.
        """
        super().__init__(*args, **kwargs)
        if backend == "open3d":
            if not OPEN3D_AVAILABLE:
                raise ValueError(
                    "You have specified the open3d backend."
                    "But open3d is not installed on this system!"
                )
            self.visualization_backend: PointCloudVisualizerBackend = (
                Open3DVisualizationBackend(
                    class_color_mapping=class_color_mapping,
                    instance_color_mapping=instance_color_mapping,
                )
            )
        else:
            raise ValueError(f"Unknown Point Visualization Backend {backend}")

        self.current_scene_idx: int | None = None
        self.current_scene: Scene3D | None = None

    def process_single(
        self,
        points_xyz: NDArrayF64,
        semantics: NDArrayI64 | None = None,
        instances: NDArrayI64 | None = None,
        colors: NDArrayF64 | None = None,
        scene_index: NDArrayI64 | int | None = None,
    ) -> None:
        """Processes data and adds it to the visualizer.

        Args:
            points_xyz: xyz coordinates of the points shape [B, N, 3]
            semantics: semantic ids of the points shape [B, N, 1]
            instances: instance ids of the points shape [B, N, 1]
            colors: colors of the points shape [B, N,3] and ranging from  [0,1]
            scene_index: Scene index for visualization of shape [B, 1].
                This allows to plot multiple predictions in the same scene
                if e.g. for memory reasons it had to be split up in multiple
                channels..

        Raises:
            ValueError: If shapes of the arrays missmatch.
        """
        # Load correct scene
        if scene_index is None:
            # No scene index given. Create new scene for each call
            self.current_scene = self.visualization_backend.create_new_scene()
        else:
            # Scene index given, check if we should update given scene
            # or create a new one
            new_scene_idx = (
                scene_index
                if isinstance(scene_index, int)
                else scene_index.item()
            )
            if (
                self.current_scene_idx is None
                or self.current_scene_idx != new_scene_idx
            ):
                self.current_scene = (
                    self.visualization_backend.create_new_scene()
                )
                self.current_scene_idx = new_scene_idx

        if self.current_scene is None:
            self.current_scene = self.visualization_backend.create_new_scene()

        # Add data to scene
        self.current_scene.add_pointcloud(
            points_xyz, colors=colors, classes=semantics, instances=instances
        )

    def process(  # pylint: disable=arguments-differ
        self,
        cur_iter: int,
        points_xyz: NDArrayF64,
        semantics: NDArrayI64 | None = None,
        instances: NDArrayI64 | None = None,
        colors: NDArrayF64 | None = None,
        scene_index: NDArrayI64 | None = None,
    ) -> None:
        """Processes a batch of data and adds it to the visualizer.

        Args:
            cur_iter: Current iteration.
            points_xyz: xyz coordinates of the points shape [N, 3]
            semantics: semantic ids of the points shape [N, 1]
            instances: instance ids of the points shape [N, 1]
            colors: colors of the points shape [N,3] and ranging from  [0,1]
            scene_index: Scene index for visualization of sape [1] or int.
                This allows to plot multiple predictions in the same scene
                if e.g. for memory reasons it had to be split up in multiple
                chunls.

        Raises:
            ValueError: If shapes of the arrays missmatch.
        """
        if self._run_on_batch(cur_iter):
            if len(points_xyz.shape) == 2:  # Data is not batched
                self.process_single(
                    points_xyz, semantics, instances, colors, scene_index
                )
            elif len(points_xyz.shape) == 3:
                for idx in range(points_xyz.shape[0]):
                    self.process_single(
                        points_xyz[idx, ...],
                        semantics[idx, ...] if semantics is not None else None,
                        instances[idx, ...] if instances is not None else None,
                        colors[idx, ...] if colors is not None else None,
                        (
                            scene_index[idx, ...]
                            if scene_index is not None
                            else None
                        ),
                    )

            else:
                raise ValueError(
                    f"Invalid shape for point data: {points_xyz.shape}"
                )

    def show(self, cur_iter: int, blocking: bool = True) -> None:
        """Shows the visualization.

        Args:
            cur_iter (int): Current iteration.
            blocking (bool): If the visualization should be blocking and wait
                for human input
        """
        self.visualization_backend.show(blocking)

    def reset(self) -> None:
        """Clears all saved data."""
        self.visualization_backend.reset()
        self.current_scene_idx = None
        self.current_scene = None

    def save_to_disk(self, cur_iter: int, output_folder: str) -> None:
        """Saves the visualization to disk."""
        if self._run_on_batch(cur_iter):
            self.visualization_backend.save_to_disk(output_folder)
