"""Video dataset loader for scalabel format."""
import json
import logging
import os
from typing import Any, Dict, List, Optional, no_type_check

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from scalabel.label.io import load
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
            assert ann.video_name is not None and ann.name is not None
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
                label.attributes = attributes

        frames.extend(imgs_anns)

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        meta.idx_to_class_mapping = dict(enumerate(cat_ids))

    return frames


@no_type_check
def load_json_to_coco(
    json_path: str, image_root: str, dataset_name: Optional[str] = None
) -> List[Dict[str, Any]]:  # type: ignore
    """Load BDD100K instances to dicts."""
    json_files = os.listdir(json_path)

    dataset_dicts = []
    instances_nonvalid_segmentation = 0
    image_id = 0
    cat_ids = []
    for json_file in json_files:
        imgs_anns = json.load(open(os.path.join(json_path, json_file), "r"))
        ins_ids = []
        for img_dict in imgs_anns:
            record = {}
            # Note: also supports pickle
            record["file_name"] = os.path.join(
                image_root, img_dict["videoName"], img_dict["name"]
            )
            record["height"] = 720  # fixed for BDD100K (720p)
            record["width"] = 1280
            record["video_id"] = img_dict["videoName"]
            record["frame_id"] = img_dict["frameIndex"]
            record["image_id"] = image_id

            objs = []
            for anno in img_dict["labels"]:
                obj = dict()

                x1 = anno["box2d"]["x1"]
                y1 = anno["box2d"]["y1"]
                x2 = anno["box2d"]["x2"]
                y2 = anno["box2d"]["y2"]
                # No + 1 for box w, h to be consistent with detectron2
                obj["bbox"] = [x1, y1, x2 - x1, y2 - y1]

                if not anno["category"] in cat_ids:
                    cat_ids.append(anno["category"])
                if not anno["id"] in ins_ids:
                    ins_ids.append(anno["id"])

                obj["category_id"] = cat_ids.index(anno["category"])
                obj["instance_id"] = ins_ids.index(anno["id"])
                obj["iscrowd"] = anno["attributes"]["Crowd"]

                segm = anno["poly2d"] if "poly2d" in anno.keys() else None
                if segm:
                    obj["segmentation"] = segm

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                objs.append(obj)
            record["annotations"] = objs
            image_id += 1
            dataset_dicts.append(record)

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = cat_ids
        meta.idx_to_class_mapping = dict(enumerate(cat_ids))

    if instances_nonvalid_segmentation > 0:
        logger.warning(
            "Filtered out %s instances without valid segmentation.",
            instances_nonvalid_segmentation,
        )
    return dataset_dicts


def register_scalabel_video_instances(  # pylint: disable=invalid-name
    name: str, metadata: Dict, json_path: str, image_root: str
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
