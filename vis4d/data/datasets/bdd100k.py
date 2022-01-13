"""Load and convert bdd100k labels to scalabel format."""

from bdd100k.common.utils import load_bdd100k_config
from bdd100k.label.to_scalabel import bdd100k_to_scalabel
from scalabel.label.io import load
from scalabel.label.typing import Dataset

from .base import BaseDatasetLoader


class BDD100K(BaseDatasetLoader):
    """BDD100K dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert BDD100K annotations to Scalabel format and prepare them."""
        assert self.annotations is not None
        bdd100k_anns = load(
            self.annotations,
            nprocs=self.num_processes,
        )
        frames = bdd100k_anns.frames
        assert self.config_path is not None
        bdd100k_cfg = load_bdd100k_config(self.config_path)

        scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
        return Dataset(frames=scalabel_frames, config=bdd100k_cfg.scalabel)
