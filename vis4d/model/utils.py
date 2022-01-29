"""Common model utilities."""
from typing import Dict, List, Optional, Union

import torch
from scalabel.label.typing import Frame

from vis4d.struct import (
    Images,
    InputSample,
    LabelInstances,
    MaskLogits,
    ModelOutput,
    SemanticMasks,
    TLabelInstance,
)


def seg_targets_to_tensor(
    images: Images, targets: LabelInstances
) -> torch.Tensor:
    """Convert Vis4D targets to torch tensor."""
    if len(targets.semantic_masks) > 1:
        # pad masks to same size for batching
        targets.semantic_masks = SemanticMasks.pad(
            targets.semantic_masks, images.tensor.shape[-2:][::-1]
        )
    return torch.stack(
        [t.to_hwc_mask() for t in targets.semantic_masks]
    ).unsqueeze(1)


def seg_tensor_to_logits(
    masks: Union[torch.Tensor, List[torch.Tensor]]
) -> List[MaskLogits]:
    """Convert segmentation tensor to list of MaskLogits."""
    if not isinstance(masks, list) and len(masks.shape) == 3:
        masks = masks.unsqueeze(0)  # pragma: no cover
    return [MaskLogits(mask) for mask in masks]


def combine_pan_outs(
    ins_outs: List[Frame], sem_outs: List[Frame]
) -> List[Frame]:
    """Combine instance and semantic segmentation outputs."""
    pan_segms = []
    for i, (ins_out, sem_out) in enumerate(zip(ins_outs, sem_outs)):
        ins_labels, sem_labels = ins_out.labels, sem_out.labels
        assert ins_labels is not None and sem_labels is not None
        pan_segms.append(Frame(name=str(i), labels=ins_labels + sem_labels))
    return pan_segms


def postprocess_predictions(
    inputs: InputSample,
    predictions: Dict[str, List[TLabelInstance]],
    clip_to_image: bool = True,
    resolve_overlap: bool = True,
) -> None:
    """Postprocess predictions."""
    for values in predictions.values():
        for inp, v in zip(inputs, values):
            size = inp.metadata[0].size
            assert size is not None
            v.postprocess(
                (size.width, size.height),
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
