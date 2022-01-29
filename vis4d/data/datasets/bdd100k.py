"""Load and convert bdd100k labels to scalabel format."""
from typing import List, Tuple

from bdd100k.common.utils import load_bdd100k_config
from bdd100k.label.to_scalabel import bdd100k_to_scalabel
from scalabel.label.io import load
from scalabel.label.typing import Config, Dataset

from .base import BaseDatasetLoader


class BDD100K(BaseDatasetLoader):
    """BDD100K dataloading class."""

    def load_config(self) -> Tuple[List[Config], Config]:
        """Load BDD100K configs."""
        cfg_paths = self._get_config_path()
        assert cfg_paths is not None
        cfgs = [
            load_bdd100k_config(cfg_path).scalabel for cfg_path in cfg_paths
        ]
        combine_cfg = Config(
            categories=[c for cfg in cfgs for c in cfg.categories]
        )
        return cfgs, combine_cfg

    def load_dataset(self) -> Dataset:
        """Convert BDD100K annotations to Scalabel format and prepare them."""
        assert self.annotations is not None
        bdd100k_anns = load(
            self.annotations,
            nprocs=self.num_processes,
        )
        frames = bdd100k_anns.frames
        cfg_paths = self._get_config_path()
        assert cfg_paths is not None
        _, metadata_cfg = self.load_config()
        bdd100k_cfg = load_bdd100k_config(cfg_paths[0])
        bdd100k_cfg.scalabel = metadata_cfg

        scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
        return Dataset(frames=scalabel_frames, config=metadata_cfg)
