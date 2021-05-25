"""Load and convert bdd100k labesl to scalabel format."""
from typing import List, Optional

from bdd100k.common.utils import load_bdd100k_config
from bdd100k.label.to_scalabel import bdd100k_to_scalabel
from scalabel.label.io import load
from scalabel.label.typing import Frame

from .scalabel import prepare_scalabel_frames


def convert_and_load_bdd100k(
    image_root: str,
    annotation_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    cfg_path: Optional[str] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Convert COCO annotations to scalabel format and prepare them."""
    assert annotation_path is not None
    bdd100k_anns = load(annotation_path)
    frames = bdd100k_anns.frames
    assert cfg_path is not None
    bdd100k_cfg = load_bdd100k_config(cfg_path)

    scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
    if prepare_frames:
        prepare_scalabel_frames(
            scalabel_frames, image_root, bdd100k_cfg.scalabel, dataset_name
        )
    return scalabel_frames
