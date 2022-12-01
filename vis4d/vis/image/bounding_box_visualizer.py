"""Vis4D box visualizer."""
from __future__ import annotations

import os
from dataclasses import dataclass

from vis4d.common.typing import (
    NDArrayF64,
    NDArrayI64,
    NDArrayNumber,
    NDArrayUI8,
)
from vis4d.vis.base import Visualizer
from vis4d.vis.image.base import CanvasBackend, ImageViewerBackend
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.vis.image.utils import preprocess_boxes, preprocess_image
from vis4d.vis.image.viewer import MatplotlibImageViewer
from vis4d.vis.util import generate_color_map


@dataclass
class DetectionBox2D:
    """Dataclass storing box informations."""

    corners: tuple[float, float, float, float]
    label: str
    color: tuple[float, float, float]


@dataclass
class DataSample:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    boxes: list[DetectionBox2D]


class BoundingBoxVisualizer(Visualizer):
    """Base visualizer class."""

    def __init__(
        self,
        n_colors: int = 50,
        class_id_mapping: dict[int, str] | None = None,
        file_type: str = "png",
        image_mode: str = "RGB",
        canvas: CanvasBackend = PillowCanvasBackend(),
        viewer: ImageViewerBackend = MatplotlibImageViewer(),
    ) -> None:
        """Creates a new Visualizer for Image and Bounding Boxes.

        Args:
            n_colors (int): How many colors should be used for the internal
                            color map
            class_id_mapping (dict[int, str]): Mapping from class id to
                                                      human readable name
            file_type (str): Desired file type
            image_mode (str): Image channel mode (RGB or BGR)
            canvas (CanvasBackend): Backend that is used to draw on images
            viewer (ImageViewerBackend): Backend that is used show images
        """
        super().__init__()
        self._samples: list[DataSample] = []
        self.color_palette = generate_color_map(n_colors)
        self.class_id_mapping = (
            class_id_mapping if class_id_mapping is not None else {}
        )
        self.file_type = file_type
        self.image_mode = image_mode
        self.canvas = canvas
        self.viewer = viewer

    def reset(self) -> None:
        """Reset visualizer for new round of evaluation."""
        self._samples = []

    def process(  # type: ignore # pylint: disable=arguments-renamed,arguments-differ,line-too-long
        self,
        images: list[NDArrayNumber],
        boxes: list[NDArrayF64],
        scores: None | list[NDArrayF64] = None,
        class_ids: None | list[NDArrayI64] = None,
        track_ids: None | list[NDArrayI64] = None,
    ) -> None:
        """Processes a batch of data.

        Use .show() or .save_to_disk() to show or save the predictions.

        Args:
            images (list[np.array]): Images to show
            boxes (list[np.array]): Predicted bounding boxe with
                        shape [N, (x1,y1,x2,y2)], where  N is the number of
                        boxes.
            scores (list[np.array]): Predicted box scores each
                                                 of shape [N]
            class_ids (list[np.array]): Predicted class ids each of
                                                    shape [N]
            track_ids (list[np.array]): Predicted track ids each of
                                                    shape [N]


        """
        for idx, image in enumerate(images):
            self.process_single_image(
                image,
                boxes[idx],
                None if scores is None else scores[idx],
                None if class_ids is None else class_ids[idx],
                None if track_ids is None else track_ids[idx],
            )

    def process_single_image(
        self,
        image: NDArrayNumber,
        boxes: NDArrayF64,
        scores: None | NDArrayF64 = None,
        class_ids: None | NDArrayI64 = None,
        track_ids: None | NDArrayI64 = None,
    ) -> None:
        """Processes a single image entry.

        Use .show() or .save_to_disk() to show or save the predictions.

        Args:
            image (np.array): Image to show
            boxes (np.array): Predicted bounding boxes with
                        shape [N, (x1,y1,x2,y2)], where  N is the number of
                        boxes.
            scores (np.array): Predicted box scores of shape [N]
            class_ids (np.array): Predicted class ids of shape [N]
            track_ids (np.array): Predicted track ids of shape [N]
        """
        img_normalized = preprocess_image(image, mode=self.image_mode)
        data_sample = DataSample(img_normalized, [])

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

    def show(self, blocking: bool = True) -> None:
        """Shows the processed images in a interactive window.

        Args:
            blocking (bool): If the visualizer should be blocking
                             i.e. wait for human input for each image
        """
        image_data = [self._draw_image(d) for d in self._samples]
        self.viewer.show_images(image_data, blocking=blocking)

    def _draw_image(self, sample: DataSample) -> NDArrayUI8:
        """Visualizes the datasample and returns is as numpy image.

        Args:
            sample (DataSample): The data sample to visualize

        Returns:
            np.array[uint8]: A image with the visualized data sample
        """
        self.canvas.create_canvas(sample.image)
        for box in sample.boxes:
            self.canvas.draw_box(box.corners, box.color)
            self.canvas.draw_text(box.corners[:2], box.label)

        return self.canvas.as_numpy_image()

    def save_to_disk(self, path_to_out_folder: str) -> None:
        """Saves the visualization to disk.

        Writes all processes samples to the output folder naming each image
        #####.<filetype> where ##### is the zero padded index of the sample

        Args:
            path_to_out_folder (str): Path to output folder.
                                      All folders in the path will be created
                                      if they do not already exist
        """
        os.makedirs(path_to_out_folder, exist_ok=True)

        for idx, sample in enumerate(self._samples):
            image_name = f"{idx:04d}.{self.file_type}"

            self.canvas.create_canvas(sample.image)

            for box in sample.boxes:
                self.canvas.draw_box(box.corners, box.color)
                self.canvas.draw_text(box.corners[:2], box.label)

            self.canvas.save_to_disk(
                os.path.join(path_to_out_folder, image_name)
            )
