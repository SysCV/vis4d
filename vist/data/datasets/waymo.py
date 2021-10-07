"""Load and convert waymo labels to scalabel format."""
import inspect
import os

from scalabel.label.io import load, load_label_config, save
from scalabel.label.typing import Dataset

from .base import BaseDatasetConfig, BaseDatasetLoader

try:
    from scalabel.label.from_waymo import from_waymo

    WAYMO_INSTALLED = True  # pragma: no cover
except NameError:
    WAYMO_INSTALLED = False


class WaymoDatasetConfig(BaseDatasetConfig):
    """Config for training/evaluation datasets."""

    input_dir: str
    output_dir: str
    save_images: bool = True
    use_lidar_labels: bool = False


class Waymo(BaseDatasetLoader):  # pragma: no cover
    """Waymo Open dataloading class."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__(cfg)
        self.cfg = WaymoDatasetConfig(**cfg.dict())  # type: WaymoDatasetConfig

    def load_dataset(self) -> Dataset:
        """Convert Waymo annotations to Scalabel format."""
        assert (
            WAYMO_INSTALLED
        ), "Using waymo dataset needs waymo open dataset reader installed!."
        assert (
            self.cfg.data_root == self.cfg.output_dir
        ), "Waymo requires conversion output path to be equal to data_root."
        cfg_path = self.cfg.config_path
        if cfg_path is None:
            cfg_path = os.path.join(
                os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
                "waymo.toml",
            )
        metadata_cfg = load_label_config(cfg_path)

        # cfg.annotations is the path to the label file in scalabel format.
        # It's an optional attribute. When passed, if the file exists load it,
        # else create it to that location
        if self.cfg.annotations:
            scalabel_anns_path = self.cfg.annotations
        else:
            scalabel_anns_path = os.path.join(
                self.cfg.output_dir, "scalabel_anns.json"
            )

        if not os.path.exists(scalabel_anns_path):
            # Read labels from tfrecords and save them to scalabel format
            frames = from_waymo(
                self.cfg.input_dir,
                self.cfg.output_dir,
                self.cfg.save_images,
                self.cfg.use_lidar_labels,
                self.cfg.num_processes,
            )
            save(scalabel_anns_path, frames)
        else:
            # Load labels from existing file
            frames = load(
                scalabel_anns_path,
                validate_frames=self.cfg.validate_frames,
                nprocs=self.cfg.num_processes,
            ).frames

        return Dataset(frames=frames, config=metadata_cfg)
