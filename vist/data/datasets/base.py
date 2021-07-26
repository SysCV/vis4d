"""Function for registering the datasets in VisT."""
import abc
import logging
import os
from typing import List, Optional

from fvcore.common.timer import Timer
from pydantic import BaseModel, validator
from scalabel.label.typing import Dataset, Frame

from vist.common.io import DataBackendConfig
from vist.common.registry import RegistryHolder
from vist.data.transforms import AugmentationConfig

logger = logging.getLogger(__name__)


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


class DataloaderConfig(BaseModel):
    """Config for dataloader."""

    image_channel_mode: str
    data_backend: DataBackendConfig = DataBackendConfig()
    categories: Optional[List[str]] = None
    skip_empty_samples: bool = False
    compute_global_instance_ids: bool = False
    transformations: Optional[List[AugmentationConfig]] = None
    ref_sampling_cfg: ReferenceSamplingConfig = ReferenceSamplingConfig()


class BaseDatasetConfig(BaseModel, extra="allow"):
    """Config for training/evaluation datasets."""

    name: str
    type: str
    data_root: str
    dataloader_cfg: DataloaderConfig
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
        timer = Timer()
        metadata_cfg, frames = self.load_dataset()
        assert metadata_cfg is not None
        add_data_path(cfg.data_root, frames)
        logger.info(
            "Loading %s takes %s seconds.",
            cfg.name,
            "{:.2f}".format(timer.seconds()),
        )
        self.metadata_cfg = metadata_cfg
        self.frames = frames

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
        if ann.video_name is not None:
            ann.url = os.path.join(data_root, ann.video_name, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.name)
