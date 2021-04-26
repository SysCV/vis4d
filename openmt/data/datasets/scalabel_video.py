"""Video dataset loader for scalabel format."""

import logging
import os
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional

from bdd100k.common.utils import DEFAULT_COCO_CONFIG
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from scalabel.label.io import load
from scalabel.label.to_coco import load_coco_config
from scalabel.label.typing import Frame

logger = logging.getLogger(__name__)


def load_json(
    json_path: str, image_root: str, dataset_name: Optional[str] = None
) -> List[Frame]:
    """Load Scalabel frames from json."""
    frames = load(json_path, nprocs=cpu_count())
    cat_ids = []  # type: List[str]
    ins_ids = []  # type: List[str]
    name_mapping, ignore_mapping = None, None
    if dataset_name is not None and "bdd100k" in dataset_name:
        cat_dicts, name_mapping, ignore_mapping = load_coco_config(
            "box_track", DEFAULT_COCO_CONFIG, False
        )
        cat_ids = [cat["name"] for cat in cat_dicts]

    for ann in frames:
        # add filename, category and instance id (integer)
        assert ann.name is not None and ann.labels is not None
        if ann.video_name is not None:
            ann.url = os.path.join(image_root, ann.video_name, ann.name)
        else:
            ann.url = os.path.join(image_root, ann.name)

        for j in range(len(ann.labels)):
            label = ann.labels[j]
            assert label.category is not None
            if not label.category in cat_ids:
                cat_ids.append(label.category)
            if not label.id in ins_ids:
                ins_ids.append(label.id)

            category = label.category
            ignore = False
            if name_mapping is not None:
                category = name_mapping.get(label.category, label.category)
            if ignore_mapping is not None and category in ignore_mapping:
                category = ignore_mapping[category]
                ignore = True
            label.category = category

            # parse category and track id to integer
            attrs = dict(
                category_id=cat_ids.index(label.category),
                instance_id=ins_ids.index(label.id),
                ignore=ignore,
            )

            if label.attributes is None:
                label.attributes = attrs  # type: ignore # pragma: no cover
            else:
                label.attributes.update(attrs)

    if dataset_name is not None:  # pragma: no cover
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = cat_ids
        meta.idx_to_class_mapping = dict(enumerate(cat_ids))

    return frames


def register_scalabel_video_instances(  # type: ignore # pylint: disable=invalid-name, line-too-long
    name: str,
    metadata: Dict[str, Any],
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
    MetadataCatalog.get(name).set(
        json_path=json_path,
        image_root=image_root,
        evaluator_type="tracking",
        **metadata
    )
