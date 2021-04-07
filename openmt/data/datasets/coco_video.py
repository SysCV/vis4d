"""Load and register a video dataset in coco format."""
import contextlib
import io
import logging
import os
from typing import Any, Dict, List, Optional

import pycocotools.mask as mask_util
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)


def load_coco_json(  # type: ignore
    json_file: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    extra_annotation_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load a json file with COCO's instances annotation format.

    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances
        annotation format.
        image_root (str): the directory where the images in this json file
        exists.
        dataset_name (str or None): the name of the dataset (e.g.,
        coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this
            dataset.
            * Map the category ids into a contiguous range (needed by
            standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata
              associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that
        should also be
            loaded into the dataset dict (besides "iscrowd", "bbox",
            "keypoints",
            "category_id", "segmentation"). The values for these keys will
            be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts
        format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when
        `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Raises:
        KeyError: If unrecognized categories are encountered.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds()
            )
        )

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted. Ignore
        # mypy error: https://github.com/python/mypy/issues/9656
        thing_classes = [
            c["name"]
            for c in sorted(cats, key=lambda x: x["id"])  # type: ignore
        ]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """Category ids in annotations are not in [1, #categories]!
                     We'll apply a mapping for you."""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            "%s contains %s annotations, but only %s of them match to images "
            "in the file.",
            json_file,
            total_num_anns,
            total_num_valid_anns,
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded %s images in COCO format from %s", len(imgs_anns), json_file
    )

    dataset_dicts = []

    ann_keys = [
        "iscrowd",
        "bbox",
        "keypoints",
        "category_id",
        "instance_id",
    ] + (extra_annotation_keys or [])

    instances_nonvalid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["video_id"] = img_dict["video_id"]
        record["frame_id"] = img_dict["frame_id"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation
            # file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation
            # files
            # actually contains bugs that, together with certain ways of
            # using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly
                        for poly in segm
                        if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        instances_nonvalid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating
                        # points in [0, H or W],
                        # but keypoint coordinates are integers in [0,
                        # H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel
                        # indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the "
                        "json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs  # type: ignore
        dataset_dicts.append(record)

    if instances_nonvalid_segmentation > 0:
        logger.warning(
            "Filtered out %s instances without valid segmentation. ",
            instances_nonvalid_segmentation,
        )
    return dataset_dicts


def register_coco_video_instances(  # type: ignore
    name: str, metadata: Dict[str, Any], json_file: str, image_root: str
) -> None:
    """Register a dataset in COCO's json annotation format for tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        **metadata,
    )
