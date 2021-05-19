"""Load MOTChallenge format dataset into Scalabel format."""
import inspect
import logging
import os
from typing import List, Optional

from fvcore.common.timer import Timer
from scalabel.label.from_mot import from_mot
from scalabel.label.io import load_label_config
from scalabel.label.typing import Frame

from .scalabel import prepare_scalabel_frames

logger = logging.getLogger(__name__)


def convert_and_load_motchallenge(
    image_root: str,
    annotation_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    cfg_path: Optional[str] = None,
    prepare_frames: bool = True,
) -> List[Frame]:  # pragma: no cover
    """Convert MOTChallenge annotations to scalabel format and prepare them."""
    assert (
        image_root == annotation_path
    ), "MOTChallenge format requires images and annotations in the same path."
    timer = Timer()
    if cfg_path is None:
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
            "motchallenge.toml",
        )
    metadata_cfg = load_label_config(cfg_path)

    frames = from_mot(annotation_path)
    logger.info(
        "Converting %s to Scalabel format takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    if prepare_frames:
        prepare_scalabel_frames(frames, image_root, metadata_cfg, dataset_name)
    return frames
