"""Video dataset loader for coco format."""
import json
import logging
import os
from typing import List, Optional

from fvcore.common.timer import Timer
from scalabel.label.from_coco import coco_to_scalabel
from scalabel.label.io import load_label_config
from scalabel.label.typing import Frame

from .scalabel import prepare_scalabel_frames

logger = logging.getLogger(__name__)


def convert_and_load_coco(
    image_root: str,
    annotation_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    cfg_path: Optional[str] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Convert COCO annotations to scalabel format and prepare them."""
    assert annotation_path is not None
    if not os.path.exists(annotation_path) or not os.path.isfile(
        annotation_path
    ):
        raise FileNotFoundError(f"COCO json file not found: {annotation_path}")
    timer = Timer()
    coco_anns = json.load(open(annotation_path, "r"))
    logger.info(
        "Loading %s in COCO format takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    timer.reset()
    frames, metadata_cfg = coco_to_scalabel(coco_anns)
    logger.info(
        "Converting %s to Scalabel format takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    if cfg_path is not None:
        metadata_cfg = load_label_config(cfg_path)
    if prepare_frames:
        prepare_scalabel_frames(frames, image_root, metadata_cfg, dataset_name)
    return frames
