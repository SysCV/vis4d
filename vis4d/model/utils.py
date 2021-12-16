"""Common model utilities."""
from typing import Dict, List, Optional

import torch

from ..struct import InputSample, ModelOutput, TLabelInstance


def postprocess_predictions(
    inputs: InputSample,
    predictions: Dict[str, List[TLabelInstance]],
    clip_to_image: bool = True,
    resolve_overlap: bool = True,
) -> None:
    """Postprocess predictions."""
    for values in predictions.values():
        for inp, v in zip(inputs, values):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            v.postprocess(
                input_size,
                inp.images.image_sizes[0],
                clip_to_image,
                resolve_overlap,
            )


def predictions_to_scalabel(
    predictions: Dict[str, List[TLabelInstance]],
    idx_to_class: Optional[Dict[int, str]] = None,
) -> ModelOutput:
    """Convert predictions into ModelOutput (Scalabel)."""
    outputs = {}
    for key, values in predictions.items():
        outputs[key] = [
            v.to(torch.device("cpu")).to_scalabel(idx_to_class) for v in values
        ]
    return outputs
