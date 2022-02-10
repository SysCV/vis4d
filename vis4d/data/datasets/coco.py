"""Dataset loader for coco format."""
import json
import os

from scalabel.label.from_coco import coco_to_scalabel
from scalabel.label.io import load_label_config
from scalabel.label.typing import Dataset

from .base import BaseDatasetLoader


class COCO(BaseDatasetLoader):
    """COCO dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert COCO annotations to scalabel format and prepare them."""
        assert self.annotations is not None
        if not os.path.exists(self.annotations) or not os.path.isfile(
            self.annotations
        ):
            raise FileNotFoundError(
                f"COCO json file not found: {self.annotations}"
            )
        with open(self.annotations, "r", encoding="utf-8") as f:
            coco_anns = json.load(f)
        frames, metadata_cfg = coco_to_scalabel(coco_anns)
        if self.config_path is not None:
            metadata_cfg = load_label_config(self.config_path)

        return Dataset(frames=frames, config=metadata_cfg)
