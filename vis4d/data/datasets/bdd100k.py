"""BDD100K dataset."""
from __future__ import annotations

from torch.utils.data import Dataset

from .scalabel import Scalabel

bdd100k_det_map = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
}
bdd100k_track_map = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
}
bdd100k_seg_map = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "pedestrian": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
}
bdd100k_panseg_map = {
    "dynamic": 0,
    "ego vehicle": 1,
    "ground": 2,
    "static": 3,
    "parking": 4,
    "rail track": 5,
    "road": 6,
    "sidewalk": 7,
    "bridge": 8,
    "building": 9,
    "fence": 10,
    "garage": 11,
    "guard rail": 12,
    "tunnel": 13,
    "wall": 14,
    "banner": 15,
    "billboard": 16,
    "lane divider": 17,
    "parking sign": 18,
    "pole": 19,
    "polegroup": 20,
    "street light": 21,
    "traffic cone": 22,
    "traffic device": 23,
    "traffic light": 24,
    "traffic sign": 25,
    "traffic sign frame": 26,
    "terrain": 27,
    "vegetation": 28,
    "sky": 29,
    "person": 30,
    "rider": 31,
    "bicycle": 32,
    "bus": 33,
    "car": 34,
    "caravan": 35,
    "motorcycle": 36,
    "trailer": 37,
    "train": 38,
    "truck": 39,
}

from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE

if BDD100K_AVAILABLE and SCALABEL_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.label.io import load
    from scalabel.label.typing import Dataset as ScalabelData


class BDD100K(Scalabel):
    """BDD100K type dataset, based on Scalabel."""

    def _generate_mapping(self) -> ScalabelData:
        """Generate data mapping."""
        bdd100k_anns = load(self.annotation_path)
        frames = bdd100k_anns.frames
        bdd100k_cfg = load_bdd100k_config(self.config_path)
        scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
        return ScalabelData(
            frames=scalabel_frames, config=bdd100k_cfg.scalabel
        )
