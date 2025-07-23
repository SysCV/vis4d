"""BEV Bounding box 3D visualizer."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArgsType,
    ArrayLikeFloat,
    ArrayLikeInt,
    NDArrayF32,
    NDArrayUI8,
)
from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import boxes3d_to_corners, transform_boxes3d
from vis4d.op.geometry.transform import inverse_rigid_transform
from vis4d.vis.base import Visualizer
from vis4d.vis.util import generate_color_map

from .canvas import CanvasBackend, PillowCanvasBackend
from .viewer import ImageViewerBackend, MatplotlibImageViewer


@dataclass
class BEVBox:
    """Dataclass storing box informations."""

    corners: list[tuple[float, float]]
    color: tuple[int, int, int]
    track_id: int | None


@dataclass
class DataSample:
    """Dataclass storing a data sample that can be visualized."""

    name: str
    extrinsics: NDArrayF32
    sequence_name: str | None
    boxes: list[BEVBox]


class BEVBBox3DVisualizer(Visualizer):
    """BEV Bounding box 3D visualizer class."""

    def __init__(
        self,
        *args: ArgsType,
        n_colors: int = 50,
        file_type: str = "png",
        max_range: float = 60,
        scale: float = 10,
        width: int = 2,
        margin: int = 10,
        axis_mode: AxisMode = AxisMode.ROS,
        trajectory_length: int = 10,
        plot_trajectory: bool = True,
        canvas: CanvasBackend | None = None,
        viewer: ImageViewerBackend | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates a new Visualizer for BEV Image and Bounding Boxes.

        Args:
            n_colors (int): How many colors should be used for the internal
                color map. Defaults to 100.
            file_type (str): Desired file type. Defaults to "png".
            max_range (float): Maximum range (meters) of the BEV image.
                Defaults to 60.
            scale (float): Scale of the BEV image. Defaults to 10. Means that
                1m in the BEV image is 10px.
            width (int): Width of the drawn bounding boxes. Defaults to 2.
            margin (int): Margin of the BEV image. Defaults to 10.
            axis_mode (AxisMode): Axis mode for the input bboxes. Defaults to
                AxisMode.ROS (i.e. global coordinate).
            trajectory_length (int): How many past frames should be used to
                draw the trajectory. Defaults to 10.
            plot_trajectory (bool): If the trajectory should be plotted.
                Defaults to True.
            canvas (CanvasBackend): Backend that is used to draw on images. If
                None a PillowCanvasBackend is used.
            viewer (ImageViewerBackend): Backend that is used show images. If
                None a MatplotlibImageViewer is used.
        """
        super().__init__(*args, **kwargs)
        self._samples: list[DataSample] = []
        self.axis_mode = axis_mode
        self.trajectories: dict[int, list[tuple[float, float, float]]] = (
            defaultdict(list)
        )
        self.trajectory_length = trajectory_length
        self.plot_trajectory = plot_trajectory

        self.color_palette = generate_color_map(n_colors)

        self.file_type = file_type
        self.max_range = max_range
        self.scale = scale

        # Generate figure size
        self.figure_hw = (
            int(max_range * scale + margin) * 2,
            int(max_range * scale + margin) * 2,
        )

        self.width = width

        self.canvas = canvas if canvas is not None else PillowCanvasBackend()
        self.viewer = viewer if viewer is not None else MatplotlibImageViewer()

    def __repr__(self) -> str:
        """Return string representation."""
        return "BEVBBox3DVisualizer"

    def reset(self) -> None:
        """Reset visualizer."""
        self._samples.clear()

    def process(  # pylint: disable=arguments-differ
        self,
        cur_iter: int,
        sample_names: list[list[str]] | list[str],
        boxes3d: list[ArrayLikeFloat],
        extrinsics: list[ArrayLikeFloat] | ArrayLikeFloat,
        class_ids: None | list[ArrayLikeInt] = None,
        track_ids: None | list[ArrayLikeInt] = None,
        sequence_names: None | list[str] = None,
    ) -> None:
        """Processes a batch of data."""
        # Handle multi-sensor connector results from multi-sensor data dict
        if isinstance(sample_names[0], list) and isinstance(extrinsics, list):
            sample_names = sample_names[0]
            extrinsics = extrinsics[0]

        if self._run_on_batch(cur_iter):
            for batch, sample_name in enumerate(sample_names):
                self.process_single(
                    sample_name,  # type: ignore
                    boxes3d[batch],
                    extrinsics[batch],  # type: ignore
                    class_ids[batch] if class_ids is not None else None,
                    track_ids[batch] if track_ids is not None else None,
                    (
                        sequence_names[batch]
                        if sequence_names is not None
                        else None
                    ),
                )

            for tid in self.trajectories:
                if len(self.trajectories[tid]) > self.trajectory_length:
                    self.trajectories[tid].pop(0)

    def process_single(
        self,
        sample_name: str,
        boxes3d: ArrayLikeFloat,
        extrinsics: ArrayLikeFloat,
        class_ids: None | ArrayLikeInt = None,
        track_ids: None | ArrayLikeInt = None,
        sequence_name: None | str = None,
    ) -> None:
        """Process single batch."""
        boxes3d = array_to_numpy(boxes3d, n_dims=2, dtype=np.float32)
        extrinsics_np = array_to_numpy(extrinsics, n_dims=2, dtype=np.float32)
        data_sample = DataSample(
            sample_name,
            extrinsics_np,
            sequence_name,
            [],
        )

        boxes3d_lidar, boxes3d = self._get_lidar_and_global_boxes3d(
            boxes3d, extrinsics_np
        )

        corners = boxes3d_to_corners(
            boxes3d_lidar, axis_mode=AxisMode.LIDAR
        ).numpy()

        track_ids_np = array_to_numpy(track_ids, n_dims=1, dtype=np.int32)
        class_ids_np = array_to_numpy(class_ids, n_dims=1, dtype=np.int32)

        for i in range(corners.shape[0]):
            track_id = None if track_ids_np is None else int(track_ids_np[i])
            class_id = None if class_ids_np is None else int(class_ids_np[i])

            if track_id is not None:
                color = self.color_palette[track_id % len(self.color_palette)]
                self.trajectories[track_id].append(
                    tuple(boxes3d[i][:3].tolist())
                )
            elif class_id is not None:
                color = self.color_palette[class_id % len(self.color_palette)]
            else:
                color = (255, 0, 0)

            data_sample.boxes.append(
                BEVBox(
                    [tuple(pts) for pts in corners[i, :4, :2]],
                    color,
                    track_id=track_id,
                )
            )

        self._samples.append(data_sample)

    def _get_lidar_and_global_boxes3d(
        self, boxes3d: NDArrayF32, extrinsics: NDArrayF32
    ) -> tuple[Tensor, NDArrayF32]:
        """Get boxes3d in lidar and global frame."""
        if self.axis_mode == AxisMode.ROS:
            global_to_lidar = inverse_rigid_transform(
                torch.from_numpy(extrinsics)
            )

            boxes3d_global = boxes3d

            boxes3d_lidar = transform_boxes3d(
                torch.from_numpy(boxes3d),
                global_to_lidar,
                source_axis_mode=self.axis_mode,
                target_axis_mode=AxisMode.LIDAR,
            )
        elif self.axis_mode == AxisMode.LIDAR:
            boxes3d_global = transform_boxes3d(
                torch.from_numpy(boxes3d),
                torch.from_numpy(extrinsics),
                source_axis_mode=self.axis_mode,
                target_axis_mode=AxisMode.ROS,
            ).numpy()

            boxes3d_lidar = torch.from_numpy(boxes3d)
        else:
            raise NotImplementedError(
                f"Axis mode {self.axis_mode} not supported"
            )
        return boxes3d_lidar, boxes3d_global

    def show(self, cur_iter: int, blocking: bool = True) -> None:
        """Shows the processed images in a interactive window.

        Args:
            cur_iter (int): Current iteration.
            blocking (bool): If the visualizer should be blocking i.e. wait for
                human input for each image. Defaults to True.
        """
        if self._run_on_batch(cur_iter):
            image_data = [self._draw_image(d) for d in self._samples]
            self.viewer.show_images(image_data, blocking=blocking)

    def _map_lidar_to_bev_image(
        self, point_x: float, point_y: float
    ) -> tuple[float, float]:
        """Maps a point from lidar frame to BEV image frame."""
        return (
            self.scale * point_x + self.figure_hw[1] // 2,
            self.scale * -point_y + self.figure_hw[0] // 2,
        )

    def _draw_image(self, sample: DataSample) -> NDArrayUI8:
        """Visualizes the datasample and returns is as numpy image.

        Args:
            sample (DataSample): The data sample to visualize.

        Returns:
            NDArrayUI8: A image with the visualized data sample.
        """
        self.canvas.create_canvas(image_hw=self.figure_hw)

        img_center = self._map_lidar_to_bev_image(0, 0)

        # Mark range every 10m
        for i in range(int(self.max_range / 10), 0, -1):
            distance = int(10 * self.scale * i)
            grey_level = 140 + i * 10
            self.canvas.draw_circle(
                img_center, (grey_level, grey_level, grey_level), distance
            )

            self.canvas.draw_text(
                (img_center[0] + distance - 25, img_center[1]),
                f"{10 * i} m",
                color=(0, 0, 0),
            )

        # Draw ego car
        self.canvas.draw_rotated_box(
            [
                (img_center[0] - self.scale, img_center[1] - self.scale * 2),
                (img_center[0] + self.scale, img_center[1] - self.scale * 2),
                (img_center[0] - self.scale, img_center[1] + self.scale * 2),
                (img_center[0] + self.scale, img_center[1] + self.scale * 2),
            ],
            (0, 0, 0),
            self.width,
        )

        global_to_lidar = inverse_rigid_transform(
            torch.from_numpy(sample.extrinsics)
        ).numpy()

        for box in sample.boxes:
            corners = [
                self._map_lidar_to_bev_image(pts[0], pts[1])
                for pts in box.corners
            ]
            self.canvas.draw_rotated_box(corners, box.color, self.width)

            if self.plot_trajectory:
                assert (
                    box.track_id is not None
                ), "Track id must be set to plot trajectory."

                trajectory = self.trajectories[box.track_id]
                for center in trajectory:
                    # Move global center to current lidar frame
                    center_lidar = np.dot(global_to_lidar, [*center, 1])[:3]

                    bev_center = self._map_lidar_to_bev_image(
                        center_lidar[0], center_lidar[1]
                    )

                    self.canvas.draw_circle(
                        bev_center, box.color, self.width * 2
                    )

        return self.canvas.as_numpy_image()

    def save_to_disk(self, cur_iter: int, output_folder: str) -> None:
        """Saves the visualization to disk.

        Writes all processes samples to the output folder naming each image
        <sample.image_name>.<filetype>.

        Args:
            cur_iter (int): Current iteration.
            output_folder (str): Folder where the output should be written.
        """
        if self._run_on_batch(cur_iter):
            for sample in self._samples:
                output_dir = output_folder
                sample_name = f"{sample.name}.{self.file_type}"

                self._draw_image(sample)

                if sample.sequence_name is not None:
                    output_dir = os.path.join(output_dir, sample.sequence_name)

                output_dir = os.path.join(output_dir, "BEV")

                os.makedirs(output_dir, exist_ok=True)
                self.canvas.save_to_disk(os.path.join(output_dir, sample_name))
