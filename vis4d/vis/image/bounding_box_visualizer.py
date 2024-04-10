"""Bounding box visualizer."""

from __future__ import annotations

import os
from dataclasses import dataclass

from vis4d.common.typing import (
    ArgsType,
    ArrayLike,
    ArrayLikeFloat,
    ArrayLikeInt,
    NDArrayUI8,
)
from vis4d.vis.base import Visualizer
from vis4d.vis.util import generate_color_map

from .canvas import CanvasBackend, PillowCanvasBackend
from .util import preprocess_boxes, preprocess_image
from .viewer import ImageViewerBackend, MatplotlibImageViewer


@dataclass
class DetectionBox2D:
    """Dataclass storing box informations."""

    corners: tuple[float, float, float, float]
    label: str
    color: tuple[int, int, int]


@dataclass
class DataSample:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    image_name: str
    boxes: list[DetectionBox2D]


class BoundingBoxVisualizer(Visualizer):
    """Bounding box visualizer class."""

    def __init__(
        self,
        *args: ArgsType,
        n_colors: int = 50,
        class_id_mapping: dict[int, str] | None = None,
        file_type: str = "png",
        canvas: CanvasBackend = PillowCanvasBackend(),
        viewer: ImageViewerBackend = MatplotlibImageViewer(),
        **kwargs: ArgsType,
    ) -> None:
        """Creates a new Visualizer for Image and Bounding Boxes.

        Args:
            n_colors (int): How many colors should be used for the internal
                            color map
            class_id_mapping (dict[int, str]): Mapping from class id to
                                                      human readable name
            file_type (str): Desired file type
            canvas (CanvasBackend): Backend that is used to draw on images
            viewer (ImageViewerBackend): Backend that is used show images
        """
        super().__init__(*args, **kwargs)
        self._samples: list[DataSample] = []
        self.color_palette = generate_color_map(n_colors)
        self.class_id_mapping = (
            class_id_mapping if class_id_mapping is not None else {}
        )
        self.file_type = file_type
        self.canvas = canvas
        self.viewer = viewer

    def reset(self) -> None:
        """Reset visualizer."""
        self._samples.clear()

    def process(  # type: ignore # pylint: disable=arguments-differ
        self,
        cur_iter: int,
        images: list[ArrayLike],
        image_names: list[str],
        boxes: list[ArrayLikeFloat],
        scores: None | list[ArrayLikeFloat] = None,
        class_ids: None | list[ArrayLikeInt] = None,
        track_ids: None | list[ArrayLikeInt] = None,
    ) -> None:
        """Processes a batch of data.

        Args:
            cur_iter (int): Current iteration.
            images (list[ArrayLike]): Images to show.
            image_names (list[str]): Image names.
            boxes (list[ArrayLikeFloat]): List of predicted bounding boxes with
                shape [N, (x1, y1, x2, y2)], where  N is the number of boxes.
            scores (None | list[ArrayLikeFloat], optional): List of predicted
                box scores each of shape [N]. Defaults to None.
            class_ids (None | list[ArrayLikeInt], optional): List of predicted
                class ids each of shape [N]. Defaults to None.
            track_ids (None | list[ArrayLikeInt], optional): List of predicted
                track ids each of shape [N]. Defaults to None.
        """
        if self._run_on_batch(cur_iter):
            for idx, image in enumerate(images):
                self.process_single_image(
                    image,
                    image_names[idx],
                    boxes[idx],
                    None if scores is None else scores[idx],
                    None if class_ids is None else class_ids[idx],
                    None if track_ids is None else track_ids[idx],
                )

    def process_single_image(
        self,
        image: ArrayLike,
        image_name: str,
        boxes: ArrayLikeFloat,
        scores: None | ArrayLikeFloat = None,
        class_ids: None | ArrayLikeInt = None,
        track_ids: None | ArrayLikeInt = None,
    ) -> None:
        """Processes a single image entry.

        Args:
            image (ArrayLike): Image to show.
            image_name (str): Image name.
            boxes (ArrayLikeFloat): Predicted bounding boxes with shape
                [N, (x1,y1,x2,y2)], where  N is the number of boxes.
            scores (None | ArrayLikeFloat, optional): Predicted box scores of
                shape [N]. Defaults to None.
            class_ids (None | ArrayLikeInt, optional): Predicted class ids of
                shape [N]. Defaults to None.
            track_ids (None | ArrayLikeInt, optional): Predicted track ids of
                shape [N]. Defaults to None.
        """
        img_normalized = preprocess_image(image, mode=self.image_mode)
        data_sample = DataSample(img_normalized, image_name, [])

        for corners, label, color in zip(
            *preprocess_boxes(
                boxes,
                scores,
                class_ids,
                track_ids,
                self.color_palette,
                self.class_id_mapping,
            )
        ):
            data_sample.boxes.append(
                DetectionBox2D(
                    corners=(corners[0], corners[1], corners[2], corners[3]),
                    label=label,
                    color=color,
                )
            )

        self._samples.append(data_sample)

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

    def _draw_image(self, sample: DataSample) -> NDArrayUI8:
        """Visualizes the datasample and returns is as numpy image.

        Args:
            sample (DataSample): The data sample to visualize.

        Returns:
            NDArrayUI8: A image with the visualized data sample.
        """
        self.canvas.create_canvas(sample.image)
        for box in sample.boxes:
            self.canvas.draw_box(box.corners, box.color)
            self.canvas.draw_text(box.corners[:2], box.label)

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
                image_name = f"{sample.image_name}.{self.file_type}"

                self.canvas.create_canvas(sample.image)

                for box in sample.boxes:
                    self.canvas.draw_box(box.corners, box.color)
                    self.canvas.draw_text(box.corners[:2], box.label)

                self.canvas.save_to_disk(
                    os.path.join(output_folder, image_name)
                )
