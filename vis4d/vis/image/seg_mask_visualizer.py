"""Segmentation mask visualizer."""

from __future__ import annotations

import os
from dataclasses import dataclass

from vis4d.common.typing import (
    ArgsType,
    ArrayLikeFloat,
    ArrayLikeInt,
    ArrayLikeUInt,
    NDArrayBool,
    NDArrayUI8,
)
from vis4d.vis.base import Visualizer
from vis4d.vis.image.canvas import CanvasBackend, PillowCanvasBackend
from vis4d.vis.image.util import preprocess_image, preprocess_masks
from vis4d.vis.image.viewer import ImageViewerBackend, MatplotlibImageViewer
from vis4d.vis.util import generate_color_map


@dataclass
class SegMask2D:
    """Dataclass storing mask information."""

    mask: NDArrayBool
    color: tuple[int, int, int]


@dataclass
class ImageWithSegMask:
    """Dataclass storing a data sample that can be visualized."""

    image: NDArrayUI8
    image_name: str
    masks: list[SegMask2D]


class SegMaskVisualizer(Visualizer):
    """Segmentation mask visualizer class."""

    def __init__(
        self,
        *args: ArgsType,
        n_colors: int = 50,
        class_id_mapping: dict[int, str] | None = None,
        file_type: str = "png",
        color_palette: list[tuple[int, int, int]] | None = None,
        canvas: CanvasBackend = PillowCanvasBackend(),
        viewer: ImageViewerBackend = MatplotlibImageViewer(),
        **kwargs: ArgsType,
    ) -> None:
        """Creates a new Visualizer for Image and Bounding Boxes.

        Args:
            n_colors (int): How many colors should be used for the color map.
            class_id_mapping (dict[int, str]): Mapping from class id to
                human readable name.
            file_type (str): Desired file type
            color_palette (list[tuple[int, int, int]]): Color palette for each
                class, in RGB format (0-255). If None, a random color palette
                with n_colors is generated automatically. Defaults to None.
            canvas (CanvasBackend): Backend that is used to draw on images
            viewer (ImageViewerBackend): Backend that is used show images
        """
        super().__init__(*args, **kwargs)
        self._samples: list[ImageWithSegMask] = []
        self.color_palette = (
            generate_color_map(n_colors)
            if color_palette is None
            else color_palette
        )
        self.class_id_mapping = (
            class_id_mapping if class_id_mapping is not None else {}
        )
        self.file_type = file_type
        self.canvas = canvas
        self.viewer = viewer

    def reset(self) -> None:
        """Reset visualizer for new round of evaluation."""
        self._samples.clear()

    def _add_masks(
        self,
        data_sample: ImageWithSegMask,
        masks: ArrayLikeUInt,
        class_ids: ArrayLikeInt | None = None,
    ) -> None:
        """Adds a mask to the current data sample.

        Args:
            data_sample (ImageWithSegMask): Data sample to add mask to.
            masks (ArrayLikeUInt): Binary masks shape [N, H, W] or [H, W].
            class_ids (NDArrayInt, optional): Class ids for each mask, with
                shape [N]. Defaults to None.
        """
        if class_ids is not None:
            assert (
                class_ids.shape[0] == masks.shape[0]  # type: ignore
            ), "The amount of masks must match the given class count!"

        for mask, color in zip(
            *preprocess_masks(masks, class_ids, self.color_palette)
        ):
            data_sample.masks.append(SegMask2D(mask=mask, color=color))

    def _draw_image(self, sample: ImageWithSegMask) -> NDArrayUI8:
        """Visualizes the datasample and returns is as numpy image.

        Args:
            sample (DataSample): The data sample to visualize.

        Returns:
            NDArrayUI8: A image with the visualized data sample.
        """
        self.canvas.create_canvas(sample.image)
        for mask in sample.masks:
            self.canvas.draw_bitmap(mask.mask, mask.color)
        return self.canvas.as_numpy_image()

    def process(  # pylint: disable=arguments-differ
        self,
        cur_iter: int,
        images: list[ArrayLikeFloat],
        image_names: list[str],
        masks: list[ArrayLikeUInt],
        class_ids: list[ArrayLikeInt] | None = None,
    ) -> None:
        """Processes a batch of data.

        Args:
            cur_iter (int): Current iteration.
            images (list[ArrayLikeFloat]): Images to show.
            image_names (list[str]): Image names.
            masks (list[ArrayLikeUInt]): Segmentation masks to show, each
                with shape [H, W] or [N, H, W]. If the shape is [H, W], the
                mask is assumed to be a semantic segmentation mask with each
                pixel being the class id. If the shape is [N, H, W], each mask
                is assumed to be a binary mask with each pixel being either 0
                or 1.
            class_ids (list[ArrayLikeInt], optional): Class ids for each mask,
                with shape [N]. If set, the masks are assumed to be binary
                masks and the length of class_ids must match the amount of
                masks. Defaults to None.
        """
        if not self._run_on_batch(cur_iter):
            return

        for idx, image in enumerate(images):
            self.process_single_image(
                image,
                image_names[idx],
                masks[idx],
                None if class_ids is None else class_ids[idx],
            )

    def process_single_image(
        self,
        image: ArrayLikeFloat,
        image_name: str,
        masks: ArrayLikeUInt,
        class_ids: ArrayLikeInt | None = None,
    ) -> None:
        """Processes a single image entry.

        Args:
            image (ArrayLikeFloat): Images to show.
            image_name (str): Name of the image.
            masks (ArrayLikeUInt): Binary masks to show, each with shape
                [N, H, W] or [H, W].
            class_ids (ArrayLikeInt, optional): Class ids for each mask, with
                shape [N]. Defaults to None.
        """
        img_normalized = preprocess_image(image, mode=self.image_mode)
        data_sample = ImageWithSegMask(img_normalized, image_name, [])
        self._add_masks(data_sample, masks, class_ids)
        self._samples.append(data_sample)

    def show(self, cur_iter: int, blocking: bool = True) -> None:
        """Shows the processed images in a interactive window.

        Args:
            cur_iter (int): Current iteration.
            blocking (bool): If the visualizer should be blocking i.e. wait for
                human input for each image
        """
        if not self._run_on_batch(cur_iter):
            return
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
        if not self._run_on_batch(cur_iter):
            return
        for sample in self._samples:
            image_name = f"{sample.image_name}.{self.file_type}"

            self.canvas.create_canvas(sample.image)
            for mask in sample.masks:
                self.canvas.draw_bitmap(mask.mask, mask.color)

            self.canvas.save_to_disk(os.path.join(output_folder, image_name))
