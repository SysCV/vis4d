"""Function for registering the datasets in Vis4D."""
import abc
import os
import pickle
from typing import Dict, List, Optional, Union

from pydantic import BaseModel
from pytorch_lightning.utilities.distributed import rank_zero_info
from scalabel.label.typing import Dataset, Frame

from vis4d.common.registry import RegistryHolder
from vis4d.common.utils.time import Timer

from ..mapper import SampleMapperConfig
from ..reference import ReferenceSamplerConfig


class BaseDatasetConfig(BaseModel, extra="allow"):
    """Config for training/evaluation datasets."""

    name: str
    type: str
    data_root: str
    sample_mapper: SampleMapperConfig = SampleMapperConfig()
    ref_sampler: ReferenceSamplerConfig = ReferenceSamplerConfig()
    annotations: Optional[str]
    attributes: Optional[
        Dict[str, Union[bool, float, str, List[float], List[str]]]
    ]
    config_path: Optional[str]
    eval_metrics: List[str] = []
    validate_frames: bool = False
    ignore_unkown_cats: bool = False
    cache_as_binary: bool = False
    num_processes: int = 4
    collect_device = "cpu"
    multi_sensor_inference: bool = True
    compute_global_instance_ids: bool = False


class BaseDatasetLoader(metaclass=RegistryHolder):
    """Interface for loading dataset to scalabel format."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__()
        self.cfg = cfg
        timer = Timer()
        if self.cfg.cache_as_binary:
            assert self.cfg.annotations is not None
            if not os.path.exists(self.cfg.annotations.rstrip("/") + ".pkl"):
                dataset = self.load_dataset()
                with open(
                    self.cfg.annotations.rstrip("/") + ".pkl", "wb"
                ) as file:
                    file.write(pickle.dumps(dataset))
            else:
                with open(
                    self.cfg.annotations.rstrip("/") + ".pkl", "rb"
                ) as file:
                    dataset = pickle.loads(file.read())
        else:
            dataset = self.load_dataset()

        assert dataset.config is not None
        add_data_path(cfg.data_root, dataset.frames)
        rank_zero_info(f"Loading {cfg.name} takes {timer.time():.2f} seconds.")
        self.metadata_cfg = dataset.config
        self.frames = dataset.frames
        self.groups = dataset.groups

    @abc.abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and possibly convert dataset to scalabel format."""
        raise NotImplementedError


def build_dataset_loader(cfg: BaseDatasetConfig) -> BaseDatasetLoader:
    """Build a dataset loader."""
    registry = RegistryHolder.get_registry(BaseDatasetLoader)
    if cfg.type in registry:
        dataset_loader = registry[cfg.type](cfg)
        assert isinstance(dataset_loader, BaseDatasetLoader)
        return dataset_loader
    raise NotImplementedError(f"Dataset type {cfg.type} not found.")


def add_data_path(data_root: str, frames: List[Frame]) -> None:
    """Add filepath to frame using data_root."""
    for ann in frames:
        assert ann.name is not None
        if ann.url is None:
            if ann.videoName is not None:
                ann.url = os.path.join(data_root, ann.videoName, ann.name)
            else:
                ann.url = os.path.join(data_root, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.url)
