"""data utils."""
import itertools
import sys
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import kornia
import numpy as np
import torch
from fvcore.common.timer import Timer
from PIL import Image
from pytorch_lightning.utilities.distributed import rank_zero_info
from scalabel.label.typing import Frame
from scalabel.label.utils import check_crowd, check_ignored
from tabulate import tabulate
from termcolor import colored

from vist.struct import NDArrayUI8

D2BoxType = Dict[str, Union[bool, float, str]]


def transform_bbox(
    trans_mat: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """Apply trans_mat (3, 3) / (B, 3, 3)  to (N, 4) / (B, N, 4) xyxy boxes."""
    assert len(trans_mat.shape) == len(
        boxes.shape
    ), "trans_mat and boxes must have same number of dimensions!"
    x1y1 = boxes[..., :2]
    x2y2 = boxes[..., 2:]
    if len(boxes.shape) == 2:
        x1y1 = x1y1.unsqueeze(0)
        x2y2 = x2y2.unsqueeze(0)
        trans_mat = trans_mat.unsqueeze(0)

    x1y1 = kornia.transform_points(trans_mat, x1y1)
    x2y2 = kornia.transform_points(trans_mat, x2y2)

    x1x2 = torch.stack((x1y1[..., 0], x2y2[..., 0]), -1)
    y1y2 = torch.stack((x1y1[..., 1], x2y2[..., 1]), -1)
    transformed_boxes = torch.stack(
        (
            x1x2.min(dim=-1)[0],
            y1y2.min(dim=-1)[0],
            x1x2.max(dim=-1)[0],
            y1y2.max(dim=-1)[0],
        ),
        -1,
    )

    if len(boxes.shape) == 2:
        transformed_boxes.squeeze(0)
    return transformed_boxes


def identity_batch_collator(  # type: ignore
    batch: List[List[Dict[str, Any]]]
) -> List[List[Dict[str, Any]]]:
    """Identity function batch collator."""
    return batch


def im_decode(im_bytes: bytes, mode: str = "RGB") -> NDArrayUI8:
    """Decode to image (numpy array, RGB) from bytes."""
    pil_img = Image.open(BytesIO(bytearray(im_bytes)))
    if mode == "BGR":
        np_img = np.array(pil_img)[..., [2, 1, 0]]  # type: NDArrayUI8
    elif mode == "RGB":
        np_img = np.array(pil_img)
    else:
        raise NotImplementedError(f"{mode} not supported for image decoding!")
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
                        ann.videoName
                        if ann.videoName is not None
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
    frames: List[Frame],
    cat_name2id: Dict[str, int],
    global_instance_ids: bool = False,
) -> Dict[str, int]:
    """Add category id and instance id to labels, return class frequencies."""
    timer = Timer()
    instance_ids = defaultdict(list)  # type: Dict[str, List[str]]
    frequencies = {cat: 0 for cat in cat_name2id}
    for frame_id, ann in enumerate(frames):
        if ann.labels is not None:
            for label in ann.labels:
                attr = {}  # type: Dict[str, Union[bool, int, float, str]]
                if label.attributes is not None:
                    attr = label.attributes

                if not check_crowd(label) and not check_ignored(label):
                    assert label.category is not None
                    frequencies[label.category] += 1
                    attr["category_id"] = cat_name2id[label.category]

                    video_name = (
                        ann.videoName
                        if ann.videoName is not None
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

    rank_zero_info(
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
    rank_zero_info(
        f"Distribution of instances among all {num_classes} categories:\n"
        + colored(table, "cyan")
    )
