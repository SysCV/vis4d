"""Dataset loader for SHIFT format."""
import json
import os

from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Category, Config, Dataset, ImageSize

from .base import BaseDatasetLoader


class SHIFTDataset(BaseDatasetLoader):
    """Scalabel dataloading class."""

    CLASS_LABELS = [
        Category(name="bicycle"),
        Category(name="car"),
        Category(name="motor"),
        Category(name="truck"),
    ]

    USE_DEPTH = True

    def load_dataset(self) -> Dataset:
        """Load Scalabel frames from json."""
        assert self.cfg.annotations is not None
        dataset = load(
            self.cfg.annotations,
            validate_frames=self.cfg.validate_frames,
            nprocs=self.cfg.num_processes,
        )
        metadata_cfg = Config(
            imageSize=ImageSize(width=1280, height=800),
            categories=CLASS_LABELS,
        )
        assert metadata_cfg is not None
        dataset.config = metadata_cfg

        if USE_DEPTH:
            self.prase_depth_map()

        return dataset

    def prase_depth_map(self):
        """Add temporal url for depth images via filename replacement."""
        for frame in dataset.frames:
            depth_url = str(frame.url)
            depth_url = depth_url.replace("data.hdf5", "depth.hdf5")
            depth_url = depth_url.replace("img_center.png", "depth.png")
            frame.attributes["depth_url"] = depth_url
