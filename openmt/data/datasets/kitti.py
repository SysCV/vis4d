"""Load and convert kitti labels to scalabel format."""
import inspect
import os
import os.path as osp

from scalabel.label.from_kitti import from_kitti
from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Dataset

from .base import BaseDatasetConfig, BaseDatasetLoader


class KITTIDatasetConfig(BaseDatasetConfig):
    """Config for training/evaluation datasets."""

    input_dir: str
    output_dir: str


class KITTI(BaseDatasetLoader):  # pragma: no cover
    """KITTI dataloading class."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__(cfg)
        self.cfg = KITTIDatasetConfig(**cfg.dict())  # type: KITTIDatasetConfig

    def load_dataset(self) -> Dataset:
        """Convert kitti annotations to Scalabel format."""
        cfg_path = self.cfg.config_path
        if cfg_path is None:
            cfg_path = os.path.join(
                os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
                "kitti.toml",
            )
        metadata_cfg = load_label_config(cfg_path)

        data_dir = osp.join(
            self.cfg.input_dir, self.cfg.data_type, self.cfg.split
        )
        file_name = f"{self.cfg.data_type}_{self.cfg.split}.json"

        if not os.path.exists(os.path.join(self.cfg.output_dir, file_name)):
            frames = from_kitti(data_dir, self.cfg.data_type)
        else:
            frames = load(
                os.path.join(self.cfg.output_dir, file_name),
                validate_frames=self.cfg.validate_frames,
                nprocs=self.cfg.num_processes,
            ).frames

        return Dataset(frames=frames, config=metadata_cfg)
