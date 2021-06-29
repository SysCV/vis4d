"""Load and convert bdd100k labels to scalabel format."""
import inspect
import os

from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Dataset

from .base import BaseDatasetConfig, BaseDatasetLoader

try:
    from scalabel.label.from_waymo import from_waymo

    WAYMO_INSTALLED = True
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

        if not os.path.exists(
            os.path.join(self.cfg.output_dir, "scalabel_anns.json")
        ):
            frames = from_waymo(
                self.cfg.input_dir,
                self.cfg.output_dir,
                self.cfg.save_images,
                self.cfg.use_lidar_labels,
                self.cfg.nproc,
            )
        else:
            frames = load(
                os.path.join(self.cfg.output_dir, "scalabel_anns.json"),
                nprocs=self.cfg.nproc,
            ).frames
        return Dataset(frames=frames, config=metadata_cfg)
