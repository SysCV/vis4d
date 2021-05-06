"""Video dataset loader for scalabel format."""

import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Union

from detectron2.data.catalog import MetadataCatalog
from detectron2.utils.comm import get_world_size
from fvcore.common.timer import Timer
from scalabel.label.io import load
from scalabel.label.typing import Frame, Label

logger = logging.getLogger(__name__)


def check_ignore(label: Label, ignore_categories: Optional[List[str]]) -> bool:
    """Check if label should be ignored."""
    ignore = False
    if ignore_categories is not None:
        if label.category in ignore_categories:
            ignore = True
    if label.attributes is not None:
        crowd = label.attributes.get("crowd", False)
        ignored_label = label.attributes.get("ignore", False)
        assert isinstance(crowd, bool)
        assert isinstance(ignored_label, bool)
        ignore = ignore or crowd or ignored_label
    return ignore


def load_scalabel(
    annotation_path: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    ignore_categories: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Load Scalabel frames from json and prepare them for openMT training."""
    frames = load_frames(annotation_path, dataset_name)
    if prepare_frames:
        prepare_scalabel_frames(
            frames, image_root, dataset_name, ignore_categories, name_mapping
        )
    return frames


def load_frames(json_path: str, dataset_name: Optional[str]) -> List[Frame]:
    """Load frames in scalabel format from json."""
    timer = Timer()
    frames = load(json_path, nprocs=min(8, cpu_count() // get_world_size()))
    logger.info(
        "Loading %s takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    return frames


def prepare_scalabel_frames(
    frames: List[Frame],
    image_root: str,
    dataset_name: Optional[str] = None,
    ignore_categories: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
) -> List[Frame]:
    """Prepare scalabel frames for openMT model training."""
    timer = Timer()
    cat_ids = []  # type: List[str]
    ins_ids = defaultdict(list)  # type: Dict[str, List[str]]
    frequencies = {cat: 0 for cat in cat_ids}
    for i, ann in enumerate(frames):
        # add filename, category and instance id (integer)
        assert ann.name is not None
        if ann.video_name is not None:
            ann.url = os.path.join(image_root, ann.video_name, ann.name)
        else:
            ann.url = os.path.join(image_root, ann.name)

        if ann.labels is not None:
            for j, label in enumerate(ann.labels):
                attr = dict()  # type: Dict[str, Union[bool, int, float, str]]

                assert label.category is not None
                if name_mapping is not None:
                    label.category = name_mapping.get(
                        label.category, label.category
                    )

                # check if label should be ignored
                attr["ignore"] = check_ignore(label, ignore_categories)

                if not attr["ignore"]:
                    # assert again because mypy will complain otherwise
                    assert label.category is not None
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
            meta.thing_classes = cat_ids
            meta.idx_to_class_mapping = dict(enumerate(cat_ids))
            meta.class_frequencies = frequencies

    return frames
