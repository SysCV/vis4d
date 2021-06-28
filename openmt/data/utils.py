"""data utils."""
import itertools
import logging
import sys
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from detectron2.structures.boxes import BoxMode
from detectron2.utils.logger import log_first_n
from fvcore.common.timer import Timer
from PIL import Image
from scalabel.label.typing import Frame, Label
from scalabel.label.utils import check_crowd, check_ignored
from tabulate import tabulate
from termcolor import colored

from openmt.struct import Boxes2D, NDArray64

D2BoxType = Dict[str, Union[bool, float, str]]
logger = logging.getLogger(__name__)


def identity_batch_collator(  # type: ignore
    batch: List[List[Dict[str, Any]]]
) -> List[List[Dict[str, Any]]]:
    """Identity function batch collator."""
    return batch


def im_decode(im_bytes: bytes) -> NDArray64:
    """Decode to image (numpy array, BGR) from bytes."""
    pil_img = Image.open(BytesIO(bytearray(im_bytes)))
    np_img = np.array(pil_img)[..., [2, 1, 0]]  # type: NDArray64
    return np_img


def instance_ids_to_global(
    frames: List[Frame], local_instance_ids: Dict[str, List[str]]
) -> None:
    """Use local (per video) instance ids to produce global ones."""
    video_names = list(local_instance_ids.keys())
    for frame_id, ann in enumerate(frames):
        if ann.labels is not None:
            for label in ann.labels:
                assert label.attributes is not None
                if not check_crowd(label) and not check_ignored(label):
                    video_name = (
                        ann.video_name
                        if ann.video_name is not None
                        else "no-video-" + str(frame_id)
                    )
                    sum_previous_vids = sum(
                        (
                            len(local_instance_ids[v])
                            for v in video_names[
                                : video_names.index(video_name)
                            ]
                        )
                    )
                    label.attributes[
                        "instance_id"
                    ] = sum_previous_vids + local_instance_ids[
                        video_name
                    ].index(
                        label.id
                    )


def prepare_labels(
    cat_name2id: Dict[str, int],
    frames: List[Frame],
    global_instance_ids: bool = False,
) -> Dict[str, int]:
    """Add category id and instance id to labels, return class frequencies."""
    timer = Timer()
    instance_ids = defaultdict(list)  # type: Dict[str, List[str]]
    frequencies = {cat: 0 for cat in cat_name2id}
    for frame_id, ann in enumerate(frames):
        if ann.labels is not None:
            for label in ann.labels:
                attr = dict()  # type: Dict[str, Union[bool, int, float, str]]
                if label.attributes is not None:
                    attr = label.attributes

                if not check_crowd(label) and not check_ignored(label):
                    assert label.category is not None
                    frequencies[label.category] += 1
                    attr["category_id"] = cat_name2id[label.category]

                    video_name = (
                        ann.video_name
                        if ann.video_name is not None
                        else "no-video-" + str(frame_id)
                    )
                    if label.id not in instance_ids[video_name]:
                        instance_ids[video_name].append(label.id)
                    attr["instance_id"] = instance_ids[video_name].index(
                        label.id
                    )

                label.attributes = attr

    if global_instance_ids:
        instance_ids_to_global(frames, instance_ids)

    logger.info(
        "Preprocessing %s labels takes %s seconds.",
        len(frames),
        "{:.2f}".format(timer.seconds()),
    )
    return frequencies


def str_decode(str_bytes: bytes, encoding: Optional[str] = None) -> str:
    """Decode to string from bytes."""
    if encoding is None:
        encoding = sys.getdefaultencoding()
    return str_bytes.decode(encoding)


def dicts_to_boxes2d(target: List[D2BoxType]) -> Boxes2D:
    """Convert d2 annotation dicts representing targets to Boxes2D."""
    if len(target) == 0:
        return Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))

    boxes = torch.tensor([t["bbox"] for t in target])
    class_ids = torch.tensor(
        [t["category_id"] for t in target], dtype=torch.long
    )
    track_ids = (
        torch.tensor([t["instance_id"] for t in target], dtype=torch.long)
        if "instance_id" in target[0]
        else None
    )
    score = torch.ones((boxes.shape[0], 1))
    return Boxes2D(torch.cat([boxes, score], -1), class_ids, track_ids)


def label_to_dict(label: Label) -> D2BoxType:
    """Convert scalabel format label to d2 readable dict."""
    assert label.box2d is not None and label.attributes is not None
    ann = dict(
        bbox=(
            label.box2d.x1,
            label.box2d.y1,
            label.box2d.x2,
            label.box2d.y2,
        ),
        bbox_mode=BoxMode.XYXY_ABS,
        category_id=label.attributes["category_id"],
    )
    if label.attributes.get("instance_id", None) is not None:
        ann["instance_id"] = label.attributes["instance_id"]
    return ann


def discard_labels_outside_set(
    dataset: List[Frame], class_set: List[str]
) -> None:
    """Discard labels outside given set of classes."""
    for frame in dataset:
        remove_anns = []
        if frame.labels is not None:
            for i, ann in enumerate(frame.labels):
                if not ann.category in class_set:
                    remove_anns.append(i)
            for i in reversed(remove_anns):
                frame.labels.pop(i)


def print_class_histogram(class_frequencies: Dict[str, int]) -> None:
    """Prints out given class frequencies."""
    class_names = list(class_frequencies.keys())
    frequencies = list(class_frequencies.values())
    num_classes = len(class_names)

    n_cols = min(6, len(class_names) * 2)

    def short_name(name: str) -> str:
        """Make long class names shorter."""
        if len(name) > 13:
            return name[:11] + ".."  # pragma: no cover
        return name

    data = list(
        itertools.chain(
            *[
                [short_name(class_names[i]), int(v)]
                for i, v in enumerate(frequencies)
            ]
        )
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (n_cols - (len(data) % n_cols)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::n_cols] for i in range(n_cols)])  # type: ignore # pylint: disable=line-too-long
    table = tabulate(
        data,  # type: ignore
        headers=["category", "#instances"] * (n_cols // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(
            num_classes
        )
        + colored(table, "cyan"),
        key="message",
    )
