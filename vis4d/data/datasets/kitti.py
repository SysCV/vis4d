"""Load and convert kitti labels to scalabel format."""
import os
import os.path as osp
from typing import Optional

from scalabel.label.from_kitti import from_kitti
from scalabel.label.io import load, save
from scalabel.label.typing import Dataset

from .base import BaseDatasetConfig, BaseDatasetLoader


class KITTIDatasetConfig(BaseDatasetConfig):
    """Config for training/evaluation datasets."""

    split: Optional[str]
    data_type: Optional[str]


class KITTI(BaseDatasetLoader):  # pragma: no cover
    """KITTI dataloading class."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__(cfg)
        self.cfg: KITTIDatasetConfig = KITTIDatasetConfig(**cfg.dict())

    def load_dataset(self) -> Dataset:
        """Convert kitti annotations to Scalabel format."""
        assert self.cfg.annotations is not None
        if not os.path.exists(self.cfg.annotations):
            assert self.cfg.data_type is not None
            assert self.cfg.split is not None
            data_dir = osp.join(
                self.cfg.data_root, self.cfg.data_type, self.cfg.split
            )
            dataset = from_kitti(data_dir, self.cfg.data_type)
            save(self.cfg.annotations, dataset)
        else:
            dataset = load(
                self.cfg.annotations,
                validate_frames=self.cfg.validate_frames,
                nprocs=self.cfg.num_processes,
            )

        return dataset
