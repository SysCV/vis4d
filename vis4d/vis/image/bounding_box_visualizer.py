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
from vis4d.vis.image.utils import preprocess_image
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
    predicted_boxes: list[DetectionBox2D]
    groundtruth_boxes: list[DetectionBox2D]


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

    def _get_box_label(
        self,
        class_id: int | None,
        score: float | None,
        track_id: int | None,
        is_gt: bool = False,
    ) -> str:
        """Gets a unique string representation for a box definition.

        Args:
            class_id (int): The class id for this box
            score (float): The confidence score
            track_id (int): The track id
            is_gt (bool): Whether or not this is a groundtruth box

        Returns:
            str: Label for this box of format
                'class_name, track_id, score%, [GT]'
        """
        labels = []
        if class_id is not None:
            labels.append(self.class_id_mapping.get(class_id, str(class_id)))
        if track_id is not None:
            labels.append(str(track_id))
        if score is not None:
            labels.append(f"{score * 100:.1f}%")
        if is_gt:
            labels.append("[GT]")

        return ", ".join(labels)

    def _add_boxes(
        self,
        data_sample: DataSample,
        boxes: NDArrayF64,
        scores: None | NDArrayF64 = None,
        class_ids: None | NDArrayI64 = None,
        track_ids: None | NDArrayI64 = None,
        is_gt: bool = False,
    ) -> None:
        """Adds a list of boxes to the given data_sample.

        Args:
            data_sample (DataSample): The current data sample to which the
                                      boxes should be added
            boxes (NDArrayF64): Boxes of shape [N, 4] where N is the number of
                                boxes and the second channel consists of
                                (x1,y1,x2,y2) box coordinates.
            scores (NDArrayF64): Scores for each box shape [N]
            class_ids (NDArrayI64): Class id for each box shape [N]
            track_ids (NDArrayI64): Track id for each box shape [N]
            is_gt (bool): Flag whether this is a ground truth box or not
        """
        target_list = (
            data_sample.predicted_boxes
            if not is_gt
            else data_sample.groundtruth_boxes
        )

        for idx in range(boxes.shape[0]):
            class_id = None if class_ids is None else class_ids[idx].item()
            score = None if scores is None else scores[idx].item()
            track_id = None if track_ids is None else track_ids[idx].item()

            if track_id is not None:
                color = self.color_palette[track_id % len(self.color_palette)]
            elif class_id is not None:
                color = self.color_palette[class_id % len(self.color_palette)]
            else:
                color = (0, 255, 0) if is_gt else (0, 0, 255)

            target_list.append(
                DetectionBox2D(
                    corners=boxes[idx, ...].tolist(),
                    label=self._get_box_label(
                        class_id, score, track_id, is_gt
                    ),
                    color=color,
                )
            )

    def _draw_image(self, sample: DataSample) -> NDArrayUI8:
        """Visualizes the datasample and returns is as numpy image.

        Args:
            sample (DataSample): The data sample to visualize

        Returns:
            np.array[uint8]: A image with the visualized data sample
        """
        self.canvas.create_canvas(sample.image)
        for box in sample.groundtruth_boxes:
            if box.color is None:
                box.color = (0, 255, 0)
            self.canvas.draw_box(box.corners, box.label, box.color)

        for box in sample.predicted_boxes:
            if box.color is None:
                box.color = (0, 0, 250)
            self.canvas.draw_box(box.corners, box.label, box.color)

        return self.canvas.as_numpy_image()

    def process(  # type: ignore # pylint: disable=arguments-renamed,arguments-differ,line-too-long
        self,
        images: list[NDArrayNumber],
        pred_boxes: list[NDArrayF64] | None = None,
        pred_scores: None | list[NDArrayF64] = None,
        pred_class_ids: None | list[NDArrayI64] = None,
        pred_track_ids: None | list[NDArrayI64] = None,
        gt_boxes: None | list[NDArrayF64] = None,
        gt_scores: None | list[NDArrayF64] = None,
        gt_class_ids: None | list[NDArrayI64] = None,
        gt_track_ids: None | list[NDArrayI64] = None,
    ) -> None:
        """Processes a batch of data.

        Use .show() or .save_to_disk() to show or save the predictions.

        Args:
            images (list[np.array]): Images to show
            pred_boxes (list[np.array]): Predicted bounding boxe with
                        shape [N, (x1,y1,x2,y2)], where  N is the number of
                        boxes.
            pred_scores (list[np.array]): Predicted box scores each
                                                 of shape [N]
            pred_class_ids (list[np.array]): Predicted class ids each of
                                                    shape [N]
            pred_track_ids (list[np.array]): Predicted track ids each of
                                                    shape [N]

            gt_boxes (list[np.array]): Ground truth bounding boxe with
                        shape [N, (x1,y1,x2,y2)], where  N is the number of
                        boxes.
            gt_scores (list[np.array]): Ground truth box scores each
                                                 of shape [N]
            gt_class_ids (list[np.array]): Ground truth class ids each of
                                                    shape [N]
            gt_track_ids (list[np.array]): Ground truth track ids each of
                                                    shape [N]
        """
        for idx, image in enumerate(images):
            self.process_single_image(
                image,
                None if pred_boxes is None else pred_boxes[idx],
                None if pred_scores is None else pred_scores[idx],
                None if pred_class_ids is None else pred_class_ids[idx],
                None if pred_track_ids is None else pred_track_ids[idx],
                None if gt_boxes is None else gt_boxes[idx],
                None if gt_scores is None else gt_scores[idx],
                None if gt_class_ids is None else gt_class_ids[idx],
                None if gt_track_ids is None else gt_track_ids[idx],
            )

    def process_single_image(
        self,
        image: NDArrayNumber,
        predicted_boxes: NDArrayF64 | None = None,
        predicted_scores: None | NDArrayF64 = None,
        predicted_class_ids: None | NDArrayI64 = None,
        predicted_track_ids: None | NDArrayI64 = None,
        gt_boxes: None | NDArrayF64 = None,
        gt_scores: None | NDArrayF64 = None,
        gt_class_ids: None | NDArrayI64 = None,
        gt_track_ids: None | NDArrayI64 = None,
    ) -> None:
        """Processes a single image entry.

        Use .show() or .save_to_disk() to show or save the predictions.

        Args:
            image (np.array): Images to show
            predicted_boxes (np.array): Predicted bounding boxe with
                        shape [N, (x1,y1,x2,y2)], where  N is the number of
                        boxes.
            predicted_scores (np.array): Predicted box scores of shape [N]
            predicted_class_ids (np.array): Predicted class ids of shape [N]
            predicted_track_ids (np.array): Predicted track ids of shape [N]

            gt_boxes (np.array): Ground truth bounding boxe with
                        shape [N, (x1,y1,x2,y2)], where  N is the number of
                        boxes.
            gt_scores (np.array): Ground truth box scores of shape [N]
            gt_class_ids (np.array): Ground truth class ids of shape [N]
            gt_track_ids (np.array): Ground truth track ids of shape [N]
        """
        img_normalized = preprocess_image(image, mode=self.image_mode)
        data_sample = DataSample(img_normalized, [], [])

        if predicted_boxes is not None:
            self._add_boxes(
                data_sample,
                predicted_boxes,
                predicted_scores,
                predicted_class_ids,
                predicted_track_ids,
                is_gt=False,
            )
        if gt_boxes is not None:
            self._add_boxes(
                data_sample,
                gt_boxes,
                gt_scores,
                gt_class_ids,
                gt_track_ids,
                is_gt=True,
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

            for box in sample.groundtruth_boxes:
                if box.color is None:
                    box.color = (0, 255, 0)
                self.canvas.draw_box(box.corners, box.label, box.color)

            for box in sample.predicted_boxes:
                if box.color is None:
                    box.color = (0, 0, 250)
                self.canvas.draw_box(box.corners, box.label, box.color)

            self.canvas.save_to_disk(
                os.path.join(path_to_out_folder, image_name)
            )
