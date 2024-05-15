"""QDTrack."""

from . import (
    qdtrack_frcnn_r50_fpn_augs_1x_bdd100k,
    qdtrack_yolox_x_25e_bdd100k,
)

# Lists of available models in BDD100K Model Zoo.
AVAILABLE_MODELS = {
    "qdtrack_frcnn_r50_fpn_augs_1x_bdd100k": (
        qdtrack_frcnn_r50_fpn_augs_1x_bdd100k
    ),
    "qdtrack_yolox_x_25e_bdd100k": qdtrack_yolox_x_25e_bdd100k,
}
