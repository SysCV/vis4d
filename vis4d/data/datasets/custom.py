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
                video_name = (
                    os.path.join(root, "").replace(source_dir, "").rstrip("/")
                )
                img_files = sorted(
                    [
                        f
                        for f in files
                        if ".jpg" in f or ".png" in f or ".jpeg" in f
                    ]
                )
                if len(img_files) == 0:
                    continue  # pragma: no cover
                for i, img_file in enumerate(img_files):
                    img = Image.open(
                        os.path.join(source_dir, video_name, img_file)
                    )
                    size = ImageSize(width=img.size[0], height=img.size[1])
                    frame = Frame(
                        name=img_file,
                        videoName=video_name,
                        frameIndex=i,
                        size=size,
                    )
                    frames.append(frame)

        assert len(frames) > 0, "Input folder didn't contain any images!"
        metadata_cfg = Config(categories=[])
        return Dataset(frames=frames, config=metadata_cfg)
