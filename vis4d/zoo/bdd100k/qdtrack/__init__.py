"""QDTrack Model Zoo."""

from . import qdtrack_bdd100k, qdtrack_yolox_bdd100k

AVAILABLE_MODELS = {
    "qdtrack_bdd100k": qdtrack_bdd100k,
    "qdtrack_yolox_bdd100k": qdtrack_yolox_bdd100k,
}
