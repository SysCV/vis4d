"""Load MOTChallenge format dataset into Scalabel format."""
import inspect
import os

from scalabel.label.from_mot import from_mot
from scalabel.label.io import load_label_config
from scalabel.label.typing import Dataset

from .base import LoadDataset


class MOTChallenge(LoadDataset):
    """Custom dataloading class."""

    def load_dataset(self) -> Dataset:  # pragma: no cover
        """Convert MOTChallenge annotations to scalabel format."""
        assert (
            self.cfg.data_root == self.cfg.annotations
        ), "MOTChallenge requires images and annotations in the same path."
        cfg_path = self.cfg.config_path
        if cfg_path is None:
            cfg_path = os.path.join(
                os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
                "motchallenge.toml",
            )
        metadata_cfg = load_label_config(cfg_path)

        frames = from_mot(self.cfg.annotations)
        return Dataset(frames=frames, config=metadata_cfg)
