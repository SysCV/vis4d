"""Video dataset loader for scalabel format."""

import logging
import os
from typing import Any, Dict, List, Optional, no_type_check

from bdd100k.common.utils import DEFAULT_COCO_CONFIG
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from scalabel.label.io import load
from scalabel.label.to_coco import load_coco_config
from scalabel.label.typing import Frame

logger = logging.getLogger(__name__)


def load_json(
    json_path: str, image_root: str, dataset_name: Optional[str] = None
) -> List[Frame]:
    """Load Scalabel frames from json."""
    frames = []
    cat_ids = []
    for json_file in os.listdir(json_path):
        imgs_anns = load(os.path.join(json_path, json_file))

        # add filename, category and instance id (integer)
        ins_ids = []
        for ann in imgs_anns:
            assert (
                ann.video_name is not None
                and ann.name is not None
                and ann.labels is not None
            )
            ann.url = os.path.join(image_root, ann.video_name, ann.name)
            for j in range(len(ann.labels)):
                label = ann.labels[j]
                assert label.category is not None
                if not label.category in cat_ids:
                    cat_ids.append(label.category)
                if not label.id in ins_ids:
                    ins_ids.append(label.id)
                # parse category and track id to integer
                attributes = dict(
                    category_id=cat_ids.index(label.category),
                    instance_id=ins_ids.index(label.id),
                )
                label.attributes = attributes  # type: ignore

        frames.extend(imgs_anns)

    if dataset_name is not None:  # pragma: no cover
        meta = MetadataCatalog.get(dataset_name)
        meta.idx_to_class_mapping = dict(enumerate(cat_ids))

    return frames


@no_type_check
def load_json_to_coco(
    json_path: str, image_root: str, dataset_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load BDD100K instances to dicts."""
    if not json_path.endswith(".json"):
        json_files = os.listdir(json_path)
        img_anns_all = [load(os.path.join(json_path, f)) for f in json_files]
    else:
        img_anns_all = [[x] for x in load(json_path)]

    dataset_dicts = []
    image_id = 0
    cat_ids = []
    name_mapping, ignore_mapping = None, None
    if "bdd100k" in dataset_name:
        cat_ids, name_mapping, ignore_mapping = load_coco_config(
            "box_track", DEFAULT_COCO_CONFIG, False
        )
        cat_ids = [cat_id["name"] for cat_id in cat_ids]

    for img_anns in img_anns_all:
        ins_ids = []
        for img_ann in img_anns:
            record = {}
            if img_ann.video_name is not None:
                file_name = os.path.join(img_ann.video_name, img_ann.name)
                record["video_id"] = img_ann.video_name
                record["frame_id"] = img_ann.frame_index
            else:
                file_name = img_ann.name

            record["file_name"] = os.path.join(image_root, file_name)
            record["height"] = 720  # fixed for BDD100K (720p)
            record["width"] = 1280
            record["image_id"] = image_id

            objs = []
            for anno in img_ann.labels:
                obj = dict()

                x1 = anno.box_2d.x1
                y1 = anno.box_2d.y1
                x2 = anno.box_2d.x2
                y2 = anno.box_2d.y2
                # No + 1 for box w, h to be consistent with detectron2
                obj["bbox"] = [x1, y1, x2 - x1, y2 - y1]

                if not anno.category in cat_ids:
                    cat_ids.append(anno.category)
                if not anno.id in ins_ids:
                    ins_ids.append(anno.id)

                category = anno.category
                if name_mapping is not None:
                    category = name_mapping.get(anno.category, anno.category)

                if ignore_mapping is not None and category in ignore_mapping:
                    category = ignore_mapping[category]
                    obj["ignore"] = True

                obj["category_id"] = cat_ids.index(category)
                obj["category_name"] = category
                obj["instance_id"] = ins_ids.index(anno.id)
                obj["iscrowd"] = anno.attributes.get("crowd", False)
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                objs.append(obj)
            record["annotations"] = objs
            image_id += 1
            dataset_dicts.append(record)

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = cat_ids
        meta.idx_to_class_mapping = dict(enumerate(cat_ids))

    return dataset_dicts


def register_scalabel_video_instances(  # type: ignore # pylint: disable=invalid-name, line-too-long
    name: str,
    metadata: Dict[str, Any],
    json_path: str,
    image_root: str,
) -> None:
    """Register a dataset in scalabel json annotation format for tracking."""
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_json_to_coco(json_path, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_path=json_path,
        image_root=image_root,
        evaluator_type="tracking",
        **metadata
    )