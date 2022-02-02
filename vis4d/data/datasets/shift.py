"""Dataset loader for SHIFT format."""
import json
import os

from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Dataset, Config, Category, ImageSize

from .base import BaseDatasetLoader


class SHIFTDataset(BaseDatasetLoader):
    """Scalabel dataloading class."""

    CLASS_LABELS = [Category(name="bicycle"), 
                    Category(name="car"), 
                    Category(name="motor"), 
                    Category(name="truck")]

    def load_dataset(self) -> Dataset:
        """Load Scalabel frames from json."""
        assert self.cfg.annotations is not None
        dataset = load(
            self.cfg.annotations,
            validate_frames=self.cfg.validate_frames,
            nprocs=self.cfg.num_processes,
        )
        metadata_cfg = Config(imageSize=ImageSize(width=1280, height=800), 
                              categories=CLASS_LABELS)
        assert metadata_cfg is not None
        dataset.config = metadata_cfg
        return dataset
