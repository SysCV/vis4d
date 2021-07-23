"""Function for registering the datasets in VisT."""
import abc
import logging
import os
from typing import List, Optional

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from pydantic import BaseModel, validator
from scalabel.label.typing import Config as MetadataConfig
from scalabel.label.typing import Dataset, Frame
from scalabel.label.utils import get_leaf_categories

from vist.common.registry import RegistryHolder

logger = logging.getLogger(__name__)


class BaseDatasetConfig(BaseModel, extra="allow"):
    """Config for training/evaluation datasets."""

    name: str
    type: str
    data_root: str
    annotations: Optional[str]
    config_path: Optional[str]
    eval_metrics: List[str] = []
    inference_sampling: str = "sample_based"
    validate_frames: bool = False
    num_processes: int = 4

    @validator("inference_sampling", check_fields=False)
    def validate_inference_sampling(  # pylint: disable=no-self-argument,no-self-use,line-too-long
        cls, value: str
    ) -> str:
        """Check inference_sampling attribute."""
        if value not in ["sample_based", "sequence_based"]:
            raise ValueError(
                "inference_sampling must be sample_based or sequence_based"
            )
        return value


class BaseDatasetLoader(metaclass=RegistryHolder):
    """Interface for loading dataset to scalabel format."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and possibly convert dataset to scalabel format."""
        raise NotImplementedError


def load_dataset(dataset_cfg: BaseDatasetConfig) -> List[Frame]:
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


def build_dataset_loader(cfg: BaseDatasetConfig) -> BaseDatasetLoader:
    """Build a dataset loader."""
    registry = RegistryHolder.get_registry(BaseDatasetLoader)
    if cfg.type in registry:
        dataset_loader = registry[cfg.type](cfg)
        assert isinstance(dataset_loader, BaseDatasetLoader)
        return dataset_loader
    raise NotImplementedError(f"Dataset type {cfg.type} not found.")


def register_dataset(dataset_cfg: BaseDatasetConfig) -> None:
    """Register a dataset."""
    DatasetCatalog.register(
        dataset_cfg.name, lambda: load_dataset(dataset_cfg)
    )


def add_data_path(data_root: str, frames: List[Frame]) -> None:
    """Add filepath to frame using data_root and frame.name."""
    for ann in frames:
        assert ann.name is not None
        if ann.video_name is not None:
            ann.url = os.path.join(data_root, ann.video_name, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.name)


def add_metadata(
    metadata_cfg: MetadataConfig, dataset_cfg: BaseDatasetConfig
) -> None:
    """Add metadata to MetadataCatalog."""
    meta = MetadataCatalog.get(dataset_cfg.name)
    if meta.get("thing_classes") is None:
        cat_name2id = {
            cat.name: i + 1
            for i, cat in enumerate(
                get_leaf_categories(metadata_cfg.categories)
            )
        }
        meta.thing_classes = list(cat_name2id.keys())
        meta.idx_to_class_mapping = {v: k for k, v in cat_name2id.items()}
        meta.metadata_cfg = metadata_cfg
        meta.annotations = dataset_cfg.annotations
        meta.data_root = dataset_cfg.data_root
        meta.cfg_path = dataset_cfg.config_path
