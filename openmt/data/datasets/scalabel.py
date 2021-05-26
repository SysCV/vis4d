"""Video dataset loader for scalabel format."""

import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Union

from detectron2.data.catalog import MetadataCatalog
from detectron2.utils.comm import get_world_size
from fvcore.common.timer import Timer
from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Config as MetadataConfig
from scalabel.label.typing import Dataset, Frame
from scalabel.label.utils import get_leaf_categories

logger = logging.getLogger(__name__)


def load_scalabel(
    image_root: str,
    annotation_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    cfg_path: Optional[str] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Load Scalabel frames from json and prepare them for openMT training."""
    assert annotation_path is not None
    dataset = load_dataset(annotation_path, dataset_name)
    frames, metadata_cfg = dataset.frames, dataset.config
    if cfg_path is not None:
        metadata_cfg = load_label_config(cfg_path)
    assert metadata_cfg is not None
    if prepare_frames:
        prepare_scalabel_frames(frames, image_root, metadata_cfg, dataset_name)
    return frames


def load_dataset(json_path: str, dataset_name: Optional[str]) -> Dataset:
    """Load frames in scalabel format from json."""
    timer = Timer()
    dataset = load(json_path, nprocs=min(8, cpu_count() // get_world_size()))
    logger.info(
        "Loading %s takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    return dataset


def prepare_scalabel_frames(
    frames: List[Frame],
    image_root: str,
    metadata_cfg: MetadataConfig,
    dataset_name: Optional[str] = None,
) -> List[Frame]:
    """Prepare scalabel frames for openMT model training."""
    timer = Timer()
    cat_name2id = {
        cat.name: i + 1
        for i, cat in enumerate(get_leaf_categories(metadata_cfg.categories))
    }
    ins_ids = defaultdict(list)  # type: Dict[str, List[str]]
    frequencies = {cat: 0 for cat in cat_name2id}
    for i, ann in enumerate(frames):
        # add filename, category and instance id (integer)
        assert ann.name is not None
        if ann.video_name is not None:
            ann.url = os.path.join(image_root, ann.video_name, ann.name)
        else:
            ann.url = os.path.join(image_root, ann.name)

        if ann.labels is not None:
            for j, label in enumerate(ann.labels):
                if not label.category in cat_name2id:
                    continue  # pragma: no cover

                attr = dict()  # type: Dict[str, Union[bool, int, float, str]]
                if label.attributes is None or not label.attributes.get(
                    "crowd", False
                ):
                    assert label.category is not None
                    frequencies[label.category] += 1
                    attr["category_id"] = cat_name2id[label.category]
                    if (
                        ann.video_name is not None
                        and label.id not in ins_ids[ann.video_name]
                    ):
                        ins_ids[ann.video_name].append(label.id)
                    attr["instance_id"] = (
                        ins_ids[ann.video_name].index(label.id)
                        if ann.video_name is not None
                        else i * 256 + j
                    )  # assumes there won't be >256 labels per frame

                if label.attributes is None:
                    label.attributes = attr  # pragma: no cover
                else:
                    label.attributes.update(attr)

    logger.info(
        "Preprocessing %s images takes %s seconds.",
        len(frames),
        "{:.2f}".format(timer.seconds()),
    )

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        if meta.get("thing_classes") is None:
            meta.thing_classes = list(cat_name2id.keys())
            meta.idx_to_class_mapping = {v: k for k, v in cat_name2id.items()}
            meta.class_frequencies = frequencies
            meta.metadata_cfg = metadata_cfg

    return frames
