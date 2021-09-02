"""Function for registering the datasets in VisT."""
import abc
import os
from typing import List, Optional

from fvcore.common.timer import Timer
from pydantic import BaseModel, validator
from pytorch_lightning.utilities.distributed import rank_zero_info
from scalabel.label.typing import Dataset, Frame

from vist.common.io import DataBackendConfig
from vist.common.registry import RegistryHolder
from vist.data.transforms import AugmentationConfig


class ReferenceSamplingConfig(BaseModel):
    """Config for customizing the sampling of reference views."""

    type: str = "uniform"
    num_ref_imgs: int = 0
    scope: int = 1
    frame_order: str = "key_first"
    skip_nomatch_samples: bool = False

    @validator("scope")
    def validate_scope(  # type: ignore # pylint: disable=no-self-argument,no-self-use, line-too-long
        cls, value: int, values
    ) -> int:
        """Check scope attribute."""
        if value != 0 and value < values["num_ref_imgs"] // 2:
            raise ValueError("Scope must be higher than num_ref_imgs / 2.")
        return value

    @validator("frame_order")
    def validate_frame_order(  # pylint: disable=no-self-argument,no-self-use
        cls, value: str
    ) -> str:
        """Check frame_order attribute."""
        if not value in ["key_first", "temporal"]:
            raise ValueError("frame_order must be key_first or temporal.")
        return value


class DataloaderConfig(BaseModel):
    """Config for dataloader."""

    data_backend: DataBackendConfig = DataBackendConfig()
    categories: Optional[List[str]] = None
    fields_to_load: List[str] = ["boxes2d"]
    skip_empty_samples: bool = False
    clip_bboxes_to_image: bool = True
    compute_global_instance_ids: bool = False
    transformations: Optional[List[AugmentationConfig]] = None
    ref_sampling: ReferenceSamplingConfig = ReferenceSamplingConfig()


class BaseDatasetConfig(BaseModel, extra="allow"):
    """Config for training/evaluation datasets."""

    name: str
    type: str
    data_root: str
    dataloader: DataloaderConfig = DataloaderConfig()
    annotations: Optional[str]
    config_path: Optional[str]
    eval_metrics: List[str] = []
    validate_frames: bool = False
    ignore_unkown_cats: bool = False
    num_processes: int = 4


class BaseDatasetLoader(metaclass=RegistryHolder):
    """Interface for loading dataset to scalabel format."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__()
        self.cfg = cfg
        timer = Timer()
        dataset = self.load_dataset()
        assert dataset.config is not None
        add_data_path(cfg.data_root, dataset.frames)
        rank_zero_info(
            "Loading %s takes %s seconds.",
            cfg.name,
            "{:.2f}".format(timer.seconds()),
        )
        self.metadata_cfg = dataset.config
        self.frames = dataset.frames

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
    """Add filepath to frame using data_root and frame.name."""
    for ann in frames:
        assert ann.name is not None
        if ann.videoName is not None:
            ann.url = os.path.join(data_root, ann.videoName, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.name)
