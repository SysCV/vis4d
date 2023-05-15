"""Vis4D segmentation mask visualizer."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    NDArrayBool,
    NDArrayI64,
    NDArrayNumber,
    NDArrayUI8,
)
from vis4d.vis.base import Visualizer
from vis4d.vis.image.base import CanvasBackend, ImageViewerBackend
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.vis.image.util import preprocess_image, preprocess_masks
from vis4d.vis.image.viewer import MatplotlibImageViewer
from vis4d.vis.util import generate_color_map


@dataclass
class SegMask2D:
    """Dataclass storing mask information."""

    mask: NDArrayBool
    color: tuple[float, float, float]


@dataclass
class ImageWithSegMask:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    masks: list[SegMask2D]


class SegMaskVisualizer(Visualizer):
    """Segmentation mask visualizer class."""

    def __init__(
        self,
        n_colors: int = 50,
        class_id_mapping: dict[int, str] | None = None,
        num_samples: int = -1,
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
        self._samples: list[ImageWithSegMask] = []
        self.color_palette = generate_color_map(n_colors)
        self.class_id_mapping = (
            class_id_mapping if class_id_mapping is not None else {}
        )
        self.num_samples = num_samples
        self.file_type = file_type
        self.image_mode = image_mode
        self.canvas = canvas
        self.viewer = viewer

    def reset(self) -> None:
        """Reset visualizer for new round of evaluation."""
        self._samples = []

    def _add_masks(
        self,
        data_sample: ImageWithSegMask,
        masks: NDArrayBool,
        class_ids: None | NDArrayI64 = None,
    ) -> None:
        """Adds a mask to the current data sample.

        Args:
            data_sample (DataSample): Data sample to add mask to
            masks (NDArrayBool): Binary masks shape [N,h,w]
            class_ids (NdArrayI64, optional): Class ids for each mask shape [N]
        """
        if class_ids is not None:
            assert (
                len(class_ids) == masks.shape[0]
            ), "The amount of masks must match the given class count!"

        for mask, color in zip(*preprocess_masks(masks, class_ids)):
            data_sample.masks.append(
                SegMask2D(mask=mask.astype(bool), color=color)
            )

    def _draw_image(self, sample: ImageWithSegMask) -> NDArrayUI8:
        """Visualizes the datasample and returns is as numpy image.

        Args:
            sample (DataSample): The data sample to visualize

        Returns:
            np.array[uint8]: A image with the visualized data sample
        """
        self.canvas.create_canvas(sample.image)
        for mask in sample.masks:
            self.canvas.draw_bitmap(mask.mask, mask.color)
        return self.canvas.as_numpy_image()

    def _to_binary_mask(self, mask: NDArrayUI8) -> NDArrayBool:
        """Converts a mask to binary masks.

        Args:
            mask (np.array): The mask to convert with shape [H, W].

        Returns:
            np.array[bool]: The binary masks with shape [N, H, W].
        """
        binary_masks = []
        for class_id in np.unique(mask):
            binary_masks.append(mask == class_id)
        return np.stack(binary_masks, axis=0)

    def process(  # pylint: disable=arguments-renamed,arguments-differ
        self,
        images: ArrayLike,
        masks: list[ArrayLike],
        class_ids: list[ArrayLike] | None = None,
    ) -> None:
        """Processes a batch of data.

        Use .show() or .save_to_disk() to show or save the predictions.

        Args:
            images (list[np.array]): Images to show
            masks (list[NDArrayBool]): Binary masks to show each shape [N,h,w]
            class_ids (list[NDArrayI64]): class ids for each mask shape [N]
        """
        images = array_to_numpy(images, None)
        masks = [array_to_numpy(mask, None, np.uint8) for mask in masks]
        if class_ids is not None:
            class_ids = [
                array_to_numpy(class_id, None, np.int)
                for class_id in class_ids
            ]
        for idx, image in enumerate(images):
            if len(self._samples) >= self.num_samples:
                break
            mask = masks[idx]
            if len(mask.shape) == 2:
                assert len(mask.shape) == 2
                mask = self._to_binary_mask(mask)
            self.process_single_image(
                image,
                mask,
                None if class_ids is None else class_ids[idx],
            )

    def process_single_image(
        self,
        image: NDArrayNumber,
        masks: NDArrayBool,
        class_ids: NDArrayI64 | None,
    ) -> None:
        """Processes a single image entry.

        Use .show() or .save_to_disk() to show or save the predictions.

        Args:
            image (np.array): Images to show
            masks (NDArrayBool): Binary masks to show each shape [N,h,w]
            class_ids (NDArrayI64, optional): Binary masks to show
                each mask of shape [h,w]
        """
        img_normalized = preprocess_image(image, mode=self.image_mode)
        data_sample = ImageWithSegMask(img_normalized, [])
        self._add_masks(data_sample, masks, class_ids)
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
            for mask in sample.masks:
                self.canvas.draw_bitmap(mask.mask, mask.color)

            self.canvas.save_to_disk(
                os.path.join(path_to_out_folder, image_name)
            )
