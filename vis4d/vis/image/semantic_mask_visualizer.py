"""Vis4D box visualizer."""
from __future__ import annotations

import os
from dataclasses import dataclass

from vis4d.common.typing import (
    NDArrayBool,
    NDArrayI64,
    NDArrayNumber,
    NDArrayUI8,
)
from vis4d.vis.base import Visualizer
from vis4d.vis.image.base import CanvasBackend, ImageViewerBackend
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.vis.image.utils import preprocess_image, preprocess_masks
from vis4d.vis.image.viewer import MatplotlibImageViewer
from vis4d.vis.util import generate_color_map


@dataclass
class SemanticMask2D:
    """Dataclass storing box informations."""

    mask: NDArrayBool
    color: tuple[float, float, float]


@dataclass
class ImageWithSemanticMask:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    masks: list[SemanticMask2D]


class SemanticMaskVisualizer(Visualizer):
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
        self._samples: list[ImageWithSemanticMask] = []
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

    def _add_masks(
        self,
        data_sample: ImageWithSemanticMask,
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
                SemanticMask2D(mask=mask.astype(bool), color=color)
            )

    def _draw_image(self, sample: ImageWithSemanticMask) -> NDArrayUI8:
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

    def process(  # type: ignore # pylint: disable=arguments-renamed,arguments-differ,line-too-long
        self,
        images: NDArrayNumber,
        masks: list[NDArrayBool],
        class_ids: list[NDArrayI64 | None] | None,
    ) -> None:
        """Processes a batch of data.

        Use .show() or .save_to_disk() to show or save the predictions.

        Args:
            images (list[np.array]): Images to show
            masks (list[NDArrayBool]): Binary masks to show each shape [N,h,w]
            class_ids (list[NDArrayI64]): class ids for each mask shape [N]
        """
        for idx, image in enumerate(images):
            self.process_single_image(
                image,
                masks[idx],
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
        data_sample = ImageWithSemanticMask(img_normalized, [])
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
