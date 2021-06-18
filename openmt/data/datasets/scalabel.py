"""Dataset loader for scalabel format."""

from multiprocessing import cpu_count

from detectron2.utils.comm import get_world_size
from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Dataset

from .base import BaseDatasetLoader


class Scalabel(BaseDatasetLoader):
    """Scalabel dataloading class."""

    def load_dataset(self) -> Dataset:
        """Load Scalabel frames from json."""
        assert self.cfg.annotations is not None
        dataset = load(
            self.cfg.annotations,
            nprocs=min(8, cpu_count() // get_world_size()),
        )
        metadata_cfg = dataset.config
        if self.cfg.config_path is not None:
            metadata_cfg = load_label_config(self.cfg.config_path)
        assert metadata_cfg is not None
        dataset.config = metadata_cfg
        return dataset
