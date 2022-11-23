"""Function interface for visualization functions."""
from __future__ import annotations

from vis4d.common.typing import NDArrayF64, NDArrayI64, NDArrayNumber
from vis4d.vis.image.bounding_box_visualizer import BoundingBoxVisualizer


def imshow_bboxes(
    image: NDArrayNumber,
    boxes: NDArrayF64 | None = None,
    scores: None | NDArrayF64 = None,
    class_ids: None | NDArrayI64 = None,
    track_ids: None | NDArrayI64 = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
) -> None:
    """Shows the bounding boxes overlayed on the given image.

    Args:
        image (NDArrayNumber): Background Image
        boxes (NDArrayF64, optional): Boxes to show. Shape [N, 4] with
                                            (x1,y1,x2,y2) as corner convention
        scores (NDArrayF64, optional): Score for each box shape [N]
        class_ids (NDArrayI64, optional): Class id for each box shape [N]
        track_ids (NDArrayI64, optional): Track id for each box shape [N]
        class_id_mapping (dict[int, str], optional): Mapping to convert
                                                    class id to class name
        n_colors (int, optional): Number of distinct colors used to color the
                                  boxes. Defaults to 50.
        image_mode (str, optional): Image channel mode (RGB or BGR).
    """
    vis = BoundingBoxVisualizer(
        n_colors=n_colors,
        class_id_mapping=class_id_mapping,
        image_mode=image_mode,
    )
    vis.process_single_image(
        image,
        predicted_boxes=boxes,
        predicted_scores=scores,
        predicted_class_ids=class_ids,
        predicted_track_ids=track_ids,
    )
    vis.show()
