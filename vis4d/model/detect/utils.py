"""Detection utility functions."""
from typing import List, Optional

from vis4d.struct import Boxes2D, InputSample, InstanceMasks


def postprocess(
    inputs: InputSample,
    detections: List[Boxes2D],
    segmentations: Optional[List[InstanceMasks]],
    clip_bboxes_to_image: bool,
) -> None:
    """Call postprocessing on the detections and instance segmentations."""
    if segmentations is not None:
        segms = segmentations
    else:
        segms = [None] * len(detections)  # type: ignore
    for inp, det, segm in zip(inputs, detections, segms):
        assert inp.metadata[0].size is not None
        input_size = (
            inp.metadata[0].size.width,
            inp.metadata[0].size.height,
        )
        det.postprocess(
            input_size,
            inp.images.image_sizes[0],
            clip_bboxes_to_image,
        )
        if segm is not None:
            segm.postprocess(input_size, inp.images.image_sizes[0], det)
