"""Semantic mask visualizer."""
from __future__ import annotations

import os
from dataclasses import dataclass

from vis4d.common.typing import (
    ArgsType,
    ArrayLike,
    ArrayLikeBool,
    ArrayLikeInt,
    NDArrayBool,
    NDArrayUI8,
)
from vis4d.vis.base import Visualizer
from vis4d.vis.image.canvas import CanvasBackend, PillowCanvasBackend
from vis4d.vis.image.util import preprocess_image, preprocess_masks
from vis4d.vis.image.viewer import ImageViewerBackend, MatplotlibImageViewer
from vis4d.vis.util import generate_color_map


@dataclass
class SemanticMask2D:
    """Dataclass storing box informations."""

    mask: NDArrayBool
    color: tuple[int, int, int]


@dataclass
class ImageWithSemanticMask:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    image_name: str
    masks: list[SemanticMask2D]


class SemanticMaskVisualizer(Visualizer):
    """Base visualizer class."""

    def __init__(
        self,
        *args: ArgsType,
        n_colors: int = 50,
        class_id_mapping: dict[int, str] | None = None,
        file_type: str = "png",
        image_mode: str = "RGB",
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
            image_mode (str): Image channel mode (RGB or BGR)
            canvas (CanvasBackend): Backend that is used to draw on images
            viewer (ImageViewerBackend): Backend that is used show images
        """
        super().__init__(*args, **kwargs)
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
        self._samples.clear()

    def _add_masks(
        self,
        data_sample: ImageWithSemanticMask,
        masks: ArrayLikeBool,
        class_ids: ArrayLikeInt | None = None,
    ) -> None:
        """Adds a mask to the current data sample.

        Args:
            data_sample (DataSample): Data sample to add mask to
            masks (ArrayLikeBool): Binary masks shape [N,h,w]
            class_ids (ArrayLikeInt | None, optional): Class ids for each mask
                shape [N]. Defaults to None.
        """
        if class_ids is not None:
            assert (
                len(class_ids) == masks.shape[0]  # type: ignore
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

    def process(  # type: ignore # pylint: disable=arguments-differ
        self,
        cur_iter: int,
        images: list[ArrayLike],
        image_names: list[str],
        masks: list[ArrayLikeBool],
        class_ids: list[ArrayLikeInt | None] | None,
    ) -> None:
        """Processes a batch of data.

        Args:
            cur_iter (int): Current iteration.
            images (list[ArrayLike]): Images to show.
            image_names (list[str]): Image names.
            masks (list[ArrayLikeBool]): Binary masks to show each shape
                [N, H, W].
            class_ids (list[ArrayLikeInt] | None, optional): Class ids for each
                mask shape [N]. Defaults to None.
        """
        if self._run_on_batch(cur_iter):
            for idx, image in enumerate(images):
                self.process_single_image(
                    image,
                    image_names[idx],
                    masks[idx],
                    None if class_ids is None else class_ids[idx],
                )

    def process_single_image(
        self,
        image: ArrayLike,
        image_name: str,
        masks: ArrayLikeBool,
        class_ids: ArrayLikeInt | None = None,
    ) -> None:
        """Processes a single image entry.

        Args:
            image (ArrayLike): Images to show.
            image_name (str): Name of the image.
            masks (ArrayLikeBool): Binary masks to show each shape
                [N, H, W].
            class_ids (ArrayLikeInt | None, optional): Binary masks to show
                each mask of shape [H, W]. Defaults to None.
        """
        img_normalized = preprocess_image(image, mode=self.image_mode)
        data_sample = ImageWithSemanticMask(img_normalized, image_name, [])
        self._add_masks(data_sample, masks, class_ids)
        self._samples.append(data_sample)

    def show(self, cur_iter: int, blocking: bool = True) -> None:
        """Shows the processed images in a interactive window.

        Args:
            cur_iter (int): Current iteration.
            blocking (bool): If the visualizer should be blocking i.e. wait for
                human input for each image
        """
        if self._run_on_batch(cur_iter):
            image_data = [self._draw_image(d) for d in self._samples]
            self.viewer.show_images(image_data, blocking=blocking)

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
                for mask in sample.masks:
                    self.canvas.draw_bitmap(mask.mask, mask.color)

                self.canvas.save_to_disk(
                    os.path.join(output_folder, image_name)
                )
