"""Load MOTChallenge format dataset into Scalabel format."""
import logging
from typing import Dict, List, Optional

from fvcore.common.timer import Timer
from scalabel.label.from_mot import from_mot
from scalabel.label.typing import Frame

from .scalabel import prepare_scalabel_frames

logger = logging.getLogger(__name__)


def convert_and_load_motchallenge(
    annotation_path: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    ignore_categories: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Convert motchallenge annotations to scalabel format and prepare them."""
    timer = Timer()

    frames = from_mot(annotation_path, image_root)
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
