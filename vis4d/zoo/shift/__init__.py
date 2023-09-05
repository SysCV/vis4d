"""BDD100K Model Zoo."""

from .faster_rcnn import (
    faster_rcnn_r50_6e_shift_all_domains,
    faster_rcnn_r50_12e_shift,
    faster_rcnn_r50_36e_shift,
)
from .mask_rcnn import (
    mask_rcnn_r50_6e_shift_all_domains,
    mask_rcnn_r50_12e_shift,
    mask_rcnn_r50_36e_shift,
)
from .semantic_fpn import (
    semantic_fpn_r50_40k_shift,
    semantic_fpn_r50_40k_shift_all_domains,
    semantic_fpn_r50_160k_shift,
    semantic_fpn_r50_160k_shift_all_domains,
)

AVAILABLE_MODELS = {
    "faster_rcnn_r50_6e_shift_all_domains": faster_rcnn_r50_6e_shift_all_domains,  # pylint: disable=line-too-long
    "faster_rcnn_r50_12e_shift": faster_rcnn_r50_12e_shift,
    "faster_rcnn_r50_36e_shift": faster_rcnn_r50_36e_shift,
    "mask_rcnn_r50_6e_shift_all_domains": mask_rcnn_r50_6e_shift_all_domains,
    "mask_rcnn_r50_12e_shift": mask_rcnn_r50_12e_shift,
    "mask_rcnn_r50_36e_shift": mask_rcnn_r50_36e_shift,
    "semantic_fpn_r50_40k_shift_all_domains": semantic_fpn_r50_40k_shift_all_domains,  # pylint: disable=line-too-long
    "semantic_fpn_r50_40k_shift": semantic_fpn_r50_40k_shift,
    "semantic_fpn_r50_160k_shift_all_domains": semantic_fpn_r50_160k_shift_all_domains,  # pylint: disable=line-too-long
    "semantic_fpn_r50_160k_shift": semantic_fpn_r50_160k_shift,
}
