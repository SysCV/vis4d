"""Function for registering the datasets in openMT."""
import abc
import logging
from typing import List

from detectron2.data.catalog import DatasetCatalog
from fvcore.common.timer import Timer
from scalabel.label.typing import Dataset, Frame

from openmt.common.registry import RegistryHolder
from openmt.config import Dataset as DatasetConfig

from ..utils import add_data_path, add_metadata

logger = logging.getLogger(__name__)


def register_dataset_instances(dataset_cfg: DatasetConfig) -> None:
    """Register a dataset."""
    DatasetCatalog.register(
        dataset_cfg.name, lambda: load_dataset(dataset_cfg)
    )


class DatasetLoader(metaclass=RegistryHolder):
    """Interface for loading dataset to scalabel format."""

    def __init__(self, cfg: DatasetConfig):
        """Init dataset loader."""
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and possibly convert dataset to scalabel format."""
        raise NotImplementedError


def load_dataset(dataset_cfg: DatasetConfig) -> List[Frame]:
    """Load a dataset into Scalabel format frames given its config."""
    timer = Timer()
    dataset = build_dataset_loader(dataset_cfg).load_dataset()
    assert dataset.config is not None
    add_data_path(dataset_cfg.data_root, dataset.frames)
    add_metadata(dataset.config, dataset_cfg)
    logger.info(
        "Loading %s takes %s seconds.",
        dataset_cfg.name,
        "{:.2f}".format(timer.seconds()),
    )
    return dataset.frames


def build_dataset_loader(cfg: DatasetConfig) -> DatasetLoader:
    """Build a dataset loader."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        dataset_loader = registry[cfg.type](cfg)
        assert isinstance(dataset_loader, DatasetLoader)
        return dataset_loader
    raise NotImplementedError(f"Dataset type {cfg.type} not found.")
