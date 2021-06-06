"""Load and convert bdd100k labelsd to scalabel format."""
from multiprocessing import cpu_count

from bdd100k.common.utils import load_bdd100k_config
from bdd100k.label.to_scalabel import bdd100k_to_scalabel
from detectron2.utils.comm import get_world_size
from scalabel.label.io import load
from scalabel.label.typing import Dataset

from .base import DatasetLoader


class BDD100K(DatasetLoader):
    """BDD100K dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert BDD100K annotations to Scalabel format and prepare them."""
        assert self.cfg.annotations is not None
        bdd100k_anns = load(
            self.cfg.annotations,
            nprocs=min(8, cpu_count() // get_world_size()),
        )
        frames = bdd100k_anns.frames
        assert self.cfg.config_path is not None
        bdd100k_cfg = load_bdd100k_config(self.cfg.config_path)

        scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
        return Dataset(frames=scalabel_frames, config=bdd100k_cfg.scalabel)
