"""Load and convert bdd100k labelsd to scalabel format."""
from multiprocessing import cpu_count

from bdd100k.common.utils import load_bdd100k_config
from bdd100k.label.to_scalabel import bdd100k_to_scalabel
from detectron2.utils.comm import get_world_size
from scalabel.label.io import load
from scalabel.label.typing import Dataset

from .base import DatasetLoader


class Waymo(DatasetLoader):
    """Waymo Open dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert Waymo annotations to Scalabel format."""
        # TODO implement waymo to scalabel
        return Dataset(frames=frames, config=config)
