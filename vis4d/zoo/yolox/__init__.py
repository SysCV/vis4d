"""YOLOX Model Zoo."""

from . import yolox_s_300e_coco, yolox_tiny_300e_coco

AVAILABLE_MODELS = {
    "yolox_s_300e_coco": yolox_s_300e_coco,
    "yolox_tiny_300e_coco": yolox_tiny_300e_coco,
}
