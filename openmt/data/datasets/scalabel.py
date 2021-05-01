"""Video dataset loader for scalabel format."""

import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Union

from bdd100k.common.utils import DEFAULT_COCO_CONFIG
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import get_world_size
from fvcore.common.timer import Timer
from scalabel.label.io import load
from scalabel.label.to_coco import load_coco_config
from scalabel.label.typing import Frame

logger = logging.getLogger(__name__)


def load_json(
    json_path: str, image_root: str, dataset_name: Optional[str] = None
) -> List[Frame]:
    """Load Scalabel frames from json."""
    timer = Timer()
    frames = load(json_path, nprocs=cpu_count() // get_world_size())
    logger.info(
        "Loading %s takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    timer.reset()

    cat_ids = []  # type: List[str]
    ins_ids = defaultdict(list)  # type: Dict[str, List[str]]
    name_mapping, ignore_mapping = None, None
    if dataset_name is not None and "bdd100k" in dataset_name:
        cat_dicts, name_mapping, ignore_mapping = load_coco_config(
            "box_track", DEFAULT_COCO_CONFIG, False
        )
        cat_ids = [cat["name"] for cat in cat_dicts]
    frequencies = {cat_id: 0 for cat_id in cat_ids}

    for i, ann in enumerate(frames):
        # add filename, category and instance id (integer)
        assert ann.name is not None
        if ann.video_name is not None:
            ann.url = os.path.join(image_root, ann.video_name, ann.name)
        else:
            ann.url = os.path.join(image_root, ann.name)

        if ann.labels is not None:
            for j, label in enumerate(ann.labels):
                assert label.category is not None
                ignore = False
                category = label.category
                if name_mapping is not None:
                    category = name_mapping.get(label.category, label.category)
                if ignore_mapping is not None and category in ignore_mapping:
                    category = ignore_mapping[category]
                    ignore = True
                label.category = category

                attr = dict(ignore=ignore)  # type: Dict[str, Union[int, bool]]

                # parse category and track id to integer
                if not label.category in cat_ids:
                    cat_ids.append(label.category)
                    frequencies[label.category] = 0
                frequencies[label.category] += 1
                attr["category_id"] = cat_ids.index(label.category)
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
                    label.attributes = attr  # type: ignore # pragma: no cover
                else:
                    label.attributes.update(attr)

    logger.info(
        "Preprocessing %s images takes %s seconds.",
        len(frames),
        "{:.2f}".format(timer.seconds()),
    )

    if dataset_name is not None:  # pragma: no cover
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = cat_ids
        meta.idx_to_class_mapping = dict(enumerate(cat_ids))
        meta.class_frequencies = frequencies

    logger.info("Dataset preparation of %s successful.", dataset_name)
    return frames


def register_scalabel_instances(
    name: str,
    json_path: str,
    image_root: str,
) -> None:
    """Register a dataset in scalabel json annotation format for tracking."""
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_json(json_path, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_path=json_path, image_root=image_root)
