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
        sub_dirs = list(os.walk(self.cfg.data_root))
        source_dir = os.path.join(self.cfg.data_root, "")  # add trailing slash
        for (root, dirs, files) in sub_dirs:
            if not dirs:
                video_name = os.path.join(root, "").replace(source_dir, "")
                img_files = sorted(
                    [f for f in files if ".jpg" in f or ".png" in f]
                )
                for i, img_file in enumerate(img_files):
                    img = Image.open(
                        os.path.join(source_dir, video_name, img_file)
                    )
                    size = ImageSize(width=img.size[0], height=img.size[1])
                    frame = Frame(
                        name=img_file,
                        video_name=video_name,
                        frame_index=i,
                        size=size,
                    )
                    frames.append(frame)

        metadata_cfg = Config(categories=[])
        return Dataset(frames=frames, config=metadata_cfg)
