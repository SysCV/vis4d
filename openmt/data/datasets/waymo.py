"""Load and convert bdd100k labels to scalabel format."""
import os
import inspect

from scalabel.label.from_waymo import from_waymo
from scalabel.label.io import load_label_config
from scalabel.label.typing import Dataset

from .base import DatasetLoader


class Waymo(DatasetLoader):
    """Waymo Open dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert Waymo annotations to Scalabel format."""
        assert (
                self.cfg.data_root == self.cfg.annotations
        ), "MOTChallenge requires images and annotations in the same path."
        cfg_path = self.cfg.config_path
        if cfg_path is None:
            cfg_path = os.path.join(
                os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
                "waymo.toml",  # TODO add to package data
            )
        metadata_cfg = load_label_config(cfg_path)

        frames = from_waymo(self.cfg.annotations)
        return Dataset(frames=frames, config=metadata_cfg)
