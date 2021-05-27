"""Dataset loader for coco format."""
import json
import os

from scalabel.label.from_coco import coco_to_scalabel
from scalabel.label.io import load_label_config
from scalabel.label.typing import Dataset

from .base import LoadDataset


class COCO(LoadDataset):
    """COCO dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert COCO annotations to scalabel format and prepare them."""
        assert self.cfg.annotations is not None
        if not os.path.exists(self.cfg.annotations) or not os.path.isfile(
            self.cfg.annotations
        ):
            raise FileNotFoundError(
                f"COCO json file not found: {self.cfg.annotations}"
            )
        coco_anns = json.load(open(self.cfg.annotations, "r"))
        frames, metadata_cfg = coco_to_scalabel(coco_anns)
        if self.cfg.config_path is not None:
            metadata_cfg = load_label_config(self.cfg.config_path)

        return Dataset(frames=frames, config=metadata_cfg)
