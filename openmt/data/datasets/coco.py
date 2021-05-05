"""Video dataset loader for coco format."""
import json
import logging
import os
from typing import Dict, List, Optional

from fvcore.common.timer import Timer
from scalabel.label.from_coco import coco_to_scalabel
from scalabel.label.typing import Frame

from .scalabel import prepare_scalabel_frames

logger = logging.getLogger(__name__)


def convert_and_load_coco(
    json_path: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    ignore_categories: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Convert COCO annotations to scalabel format and prepare them."""
    if not os.path.exists(json_path) or not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO json file not found: {json_path}")
    timer = Timer()
    coco_anns = json.load(open(json_path, "r"))
    logger.info(
        "Loading %s in COCO format takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    timer.reset()
    frames, _ = coco_to_scalabel(coco_anns)
    logger.info(
        "Converting %s to Scalabel format takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    if prepare_frames:
        prepare_scalabel_frames(
            frames, image_root, dataset_name, ignore_categories, name_mapping
        )
    return frames
