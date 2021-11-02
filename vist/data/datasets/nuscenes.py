"""Load and convert NuScenes labels to scalabel format."""
import os

from scalabel.label.io import load, load_label_config, save
from scalabel.label.typing import Dataset

from .base import BaseDatasetConfig, BaseDatasetLoader

try:
    from scalabel.label.from_nuscenes import from_nuscenes

    NUSC_INSTALLED = True  # pragma: no cover
except ImportError:
    NUSC_INSTALLED = False


class NuScenesDatasetConfig(BaseDatasetConfig):
    """Config for training/evaluation datasets."""

    version: str
    split: str
    add_non_key: bool


class NuScenes(BaseDatasetLoader):  # pragma: no cover
    """NuScenes dataloading class."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__(cfg)
        self.cfg: NuScenesDatasetConfig = NuScenesDatasetConfig(**cfg.dict())

    def load_dataset(self) -> Dataset:
        """Convert NuScenes annotations to Scalabel format."""
        assert (
            NUSC_INSTALLED
        ), "Using NuScenes dataset needs NuScenes devkit installed!."

        # cfg.annotations is the path to the label file in scalabel format.
        # if the file exists load it, else create it to that location
        assert (
            self.cfg.annotations is not None
        ), "Need a path to an annotation file to either load or create it."
        if not os.path.exists(self.cfg.annotations):
            dataset = from_nuscenes(
                self.cfg.data_root,
                self.cfg.version,
                self.cfg.split,
                self.cfg.num_processes,
                self.cfg.add_non_key,
            )
            save(self.cfg.annotations, dataset)
        else:
            # Load labels from existing file
            dataset = load(
                self.cfg.annotations,
                validate_frames=self.cfg.validate_frames,
                nprocs=self.cfg.num_processes,
            )

        if self.cfg.config_path is not None:
            dataset.config = load_label_config(self.cfg.config_path)

        return dataset
