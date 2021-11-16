"""Detection utility functions."""
from typing import List

from vis4d.struct import Boxes2D, InputSample, InstanceMasks


def postprocess(
    inputs: InputSample,
    detections: List[Boxes2D],
    segmentations: List[InstanceMasks],
    clip_bboxes_to_image: bool,
) -> None:
    """Call postprocessing on the detections and instance segmentations."""
    for inp, det, segm in zip(inputs, detections, segmentations):
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
