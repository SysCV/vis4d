import json
import logging
import os
from typing import Dict, List

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from scalabel.label.typing import Frame

logger = logging.getLogger(__name__)


def load_json(json_path, image_root) -> List[Frame]:
    """Load Scalabel frames from json."""
    json_files = os.listdir(json_path)
    for i in range(len(json_files)):
        json_files[i] = PathManager.get_local_path(json_files[i])

    dataset_dicts = []
    num_instances_without_valid_segmentation = 0
    image_id = 0
    cat_ids = set()
    for json_file in json_files:
        imgs_anns = json.load(open(os.path.join(json_path, json_file), 'r'))
        ins_ids = set()
        for img_dict in imgs_anns:
            img_dict['file_name'] = os.path.join(image_root, img_dict['video_name'], img_dict['name'])
            for i in range(len(img_dict['labels'])):
                anno = img_dict['labels'][i]
                cat_ids.add(anno['category'])
                ins_ids.add(anno['id'])
                # parse category and track id to integer
                img_dict['labels'][i]["category_id"] = list(cat_ids).index(anno['category'])
                img_dict['labels'][i]["instance_id"] = list(ins_ids).index(anno['id'])

            image_id += 1
            dataset_dicts.append(Frame(**img_dict))

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def load_json_to_coco(json_path, image_root, dataset_name=None):
    """Load BDD instances to dicts."""
    json_files = os.listdir(json_path)
    for i in range(len(json_files)):
        json_files[i] = PathManager.get_local_path(json_files[i])

    dataset_dicts = []
    num_instances_without_valid_segmentation = 0
    image_id = 0
    cat_ids = set()
    for json_file in json_files:
        imgs_anns = json.load(open(os.path.join(json_path, json_file), 'r'))
        ins_ids = set()
        for img_dict in imgs_anns:
            record = {}
            # Note: also supports pickle
            record["file_name"] = os.path.join(image_root, img_dict["video_name"],
                                               img_dict["name"])
            record["height"] = 720  # fixed for BDD (720p)
            record["width"] = 1280
            record["video_id"] = img_dict["video_name"]
            record["frame_id"] = img_dict["index"]
            record["image_id"] = image_id

            objs = []
            for anno in img_dict['labels']:
                obj = dict()

                x1 = anno["box2d"]["x1"]
                y1 = anno["box2d"]["y1"]
                x2 = anno["box2d"]["x2"]
                y2 = anno["box2d"]["y2"]
                obj["bbox"] = [x1, y1, x2 - x1, y2 - y1]

                cat_ids.add(anno['category'])
                ins_ids.add(anno['id'])
                obj["category_id"] = list(cat_ids).index(anno['category'])
                obj["instance_id"] = list(ins_ids).index(anno['id'])
                obj["iscrowd"] = anno['attributes']['Crowd']

                segm = anno['poly2d'] if 'poly2d' in anno.keys() else None
                if segm:
                    obj["segmentation"] = segm  # TODO this needs bitmask parsing. Maybe more elegant to import from BDD package?

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                objs.append(obj)
            record["annotations"] = objs
            image_id += 1
            dataset_dicts.append(record)

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = list(cat_ids)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_scalabel_video_instances(name, metadata, json_path, image_root):
    """Register a dataset in scalabel json annotation format for tracking."""
    assert isinstance(name, str), name
    assert isinstance(json_path, (str, os.PathLike)), json_path
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_json_to_coco(json_path, image_root, name))  # TODO discuss on format to use (standard detectron2 vs scalabel)

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_path=json_path, image_root=image_root, evaluator_type="tracking", **metadata
    )
