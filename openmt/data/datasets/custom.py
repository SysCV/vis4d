"""Load data from custom directory."""
import os
from typing import List, Optional

from PIL import Image
from scalabel.label.typing import Config, Frame, ImageSize

from .scalabel import prepare_scalabel_frames


def convert_and_load_directory(
    image_root: str,
    annotation_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    cfg_path: Optional[str] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Convert data in directory to scalabel format."""
    assert annotation_path is None and cfg_path is None

    frames = []
    for i, img_file in enumerate(sorted(os.listdir(image_root))):
        img = Image.open(os.path.join(image_root, img_file))
        size = ImageSize(width=img.size[0], height=img.size[1])
        frame = Frame(name=img_file, video_name="", frame_index=i, size=size)
        frames.append(frame)

    metadata_cfg = Config(categories=[])
    if prepare_frames:
        prepare_scalabel_frames(frames, image_root, metadata_cfg, dataset_name)
    return frames
