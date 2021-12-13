"""Common model utilities."""
from typing import Dict, List, Optional

import torch

from ..struct import InputSample, ModelOutput, TLabelInstance


def predictions_to_scalabel(
    inputs: InputSample,
    predictions: Dict[str, List[TLabelInstance]],
    idx_to_class: Optional[Dict[int, str]] = None,
    clip_to_image: bool = True,
) -> ModelOutput:
    """Postprocess and convert predictions into ModelOutput (Scalabel)."""
    outputs = {}
    for key, values in predictions.items():
        processed_values = []
        for inp, v in zip(inputs, values):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            v.postprocess(input_size, inp.images.image_sizes[0], clip_to_image)
            processed_values.append(
                v.to(torch.device("cpu")).to_scalabel(idx_to_class)
            )
        outputs[key] = processed_values
    return outputs
