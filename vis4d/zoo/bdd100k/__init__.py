"""BDD100K Model Zoo."""

from .faster_rcnn import faster_rcnn_r50_1x_bdd100k, faster_rcnn_r50_3x_bdd100k
from .mask_rcnn import (
    mask_rcnn_r50_1x_bdd100k,
    mask_rcnn_r50_3x_bdd100k,
    mask_rcnn_r50_5x_bdd100k,
)
from .qdtrack import qdtrack_frcnn_r50_fpn_1x_bdd100k
from .semantic_fpn import (
    semantic_fpn_r50_40k_bdd100k,
    semantic_fpn_r50_80k_bdd100k,
    semantic_fpn_r101_80k_bdd100k,
)

# Lists of available models in BDD100K Model Zoo.
AVAILABLE_MODELS = {
    "faster_rcnn_r50_1x_bdd100k": faster_rcnn_r50_1x_bdd100k,
    "faster_rcnn_r50_3x_bdd100k": faster_rcnn_r50_3x_bdd100k,
    "mask_rcnn_r50_1x_bdd100k": mask_rcnn_r50_1x_bdd100k,
    "mask_rcnn_r50_3x_bdd100k": mask_rcnn_r50_3x_bdd100k,
    "mask_rcnn_r50_5x_bdd100k": mask_rcnn_r50_5x_bdd100k,
    "semantic_fpn_r50_40k_bdd100k": semantic_fpn_r50_40k_bdd100k,
    "semantic_fpn_r50_80k_bdd100k": semantic_fpn_r50_80k_bdd100k,
    "semantic_fpn_r101_80k_bdd100k": semantic_fpn_r101_80k_bdd100k,
    "qdtrack_frcnn_r50_fpn_1x_bdd100k": qdtrack_frcnn_r50_fpn_1x_bdd100k,
}
