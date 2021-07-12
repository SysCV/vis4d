"""Load data from custom directory."""
import os

from PIL import Image
from scalabel.label.typing import Config, Dataset, Frame, ImageSize

from .base import BaseDatasetLoader


class Custom(BaseDatasetLoader):
    """Custom dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert data in directory to scalabel format."""
        assert self.cfg.annotations is None and self.cfg.config_path is None

        frames = []
        for i, img_file in enumerate(sorted(os.listdir(self.cfg.data_root))):
            img = Image.open(os.path.join(self.cfg.data_root, img_file))
            size = ImageSize(width=img.size[0], height=img.size[1])
            frame = Frame(
                name=img_file, video_name="", frame_index=i, size=size
            )
            frames.append(frame)

        metadata_cfg = Config(categories=[])
        return Dataset(frames=frames, config=metadata_cfg)
