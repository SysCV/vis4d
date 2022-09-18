"""data utils."""
import copy
import itertools
import pickle
import sys
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageOps
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scalabel.label.typing import Frame, FrameGroup
from scalabel.label.utils import check_crowd, check_ignored
from tabulate import tabulate
from termcolor import colored

from vis4d.struct_to_revise import InputSample, NDArrayI64, NDArrayUI8

from ..common_to_revise.utils.distributed import (
    all_gather_object_cpu,
    all_gather_object_gpu,
)
from ..op.geometry.transform import transform_points

try:
    from cv2 import (  # pylint: disable=no-member,no-name-in-module
        COLOR_BGR2RGB,
        IMREAD_COLOR,
        cvtColor,
        imdecode,
    )

    CV2_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    CV2_INSTALLED = False


D2BoxType = Dict[str, Union[bool, float, str]]


def transform_bbox(
    trans_mat: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """Apply trans_mat (3, 3) / (B, 3, 3)  to (N, 4) / (B, N, 4) xyxy boxes."""
    assert len(trans_mat.shape) == len(
        boxes.shape
    ), "trans_mat and boxes must have same number of dimensions!"
    x1y1 = boxes[..., :2]
    x2y1 = torch.stack((boxes[..., 2], boxes[..., 1]), -1)
    x2y2 = boxes[..., 2:]
    x1y2 = torch.stack((boxes[..., 0], boxes[..., 3]), -1)

    x1y1 = transform_points(x1y1, trans_mat)
    x2y1 = transform_points(x2y1, trans_mat)
    x2y2 = transform_points(x2y2, trans_mat)
    x1y2 = transform_points(x1y2, trans_mat)

    x_all = torch.stack(
        (x1y1[..., 0], x2y2[..., 0], x2y1[..., 0], x1y2[..., 0]), -1
    )
    y_all = torch.stack(
        (x1y1[..., 1], x2y2[..., 1], x2y1[..., 1], x1y2[..., 1]), -1
    )
    transformed_boxes = torch.stack(
        (
            x_all.min(dim=-1)[0],
            y_all.min(dim=-1)[0],
            x_all.max(dim=-1)[0],
            y_all.max(dim=-1)[0],
        ),
        -1,
    )

    if len(boxes.shape) == 2:
        transformed_boxes.squeeze(0)
    return transformed_boxes


def identity_batch_collator(
    batch: List[List[InputSample]],
) -> List[List[InputSample]]:
    """Identity function batch collator."""
    return batch


def im_decode(
    im_bytes: bytes, mode: str = "RGB", backend: str = "PIL"
) -> NDArrayUI8:
    """Decode to image (numpy array, RGB) from bytes."""
    assert mode in ["BGR", "RGB"], f"{mode} not supported for image decoding!"
    if backend == "PIL":
        pil_img = Image.open(BytesIO(bytearray(im_bytes)))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode == "L":  # pragma: no cover
            # convert grayscale image to RGB
            pil_img = pil_img.convert("RGB")
        if mode == "BGR":  # pragma: no cover
            img: NDArrayUI8 = np.array(pil_img)[..., [2, 1, 0]]
        elif mode == "RGB":
            img = np.array(pil_img)
    elif backend == "cv2":  # pragma: no cover
        if not CV2_INSTALLED:
            raise ImportError(
                "Please install opencv-python to use cv2 backend!"
            )
        img_np: NDArrayUI8 = np.frombuffer(im_bytes, np.uint8)
        img = imdecode(img_np, IMREAD_COLOR)
        if mode == "RGB":
            cvtColor(img, COLOR_BGR2RGB, img)
    else:
        raise NotImplementedError(f"Image backend {backend} not known!")
    return img


def instance_ids_to_global(
    frames: Union[List[Frame], List[FrameGroup]],
    local_instance_ids: Dict[str, List[str]],
) -> None:
    """Use local (per video) instance ids to produce global ones."""
    video_names = list(local_instance_ids.keys())
    for frame_id, ann in enumerate(frames):
        if ann.labels is None:  # pragma: no cover
            continue
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
                        for v in video_names[: video_names.index(video_name)]
                    )
                )
                label.attributes[
                    "instance_id"
                ] = sum_previous_vids + local_instance_ids[video_name].index(
                    label.id
                )


def prepare_labels(
    frames: Union[List[Frame], List[FrameGroup]],
    cat_names: List[str],
    global_instance_ids: bool = False,
) -> Dict[str, int]:
    """Add category id and instance id to labels, return class frequencies."""
    instance_ids: Dict[str, List[str]] = defaultdict(list)
    frequencies = {cat: 0 for cat in cat_names}
    for frame_id, ann in enumerate(frames):
        if ann.labels is not None:
            for label in ann.labels:
                attr: Dict[str, Union[bool, int, float, str]] = {}
                if label.attributes is not None:
                    attr = label.attributes

                if not check_crowd(label) and not check_ignored(label):
                    assert label.category is not None
                    frequencies[label.category] += 1
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
    if len(class_frequencies) == 0:  # pragma: no cover
        return

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
    total_num_instances = sum(data[1::2])  # type: ignore
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


def check_attributes(
    frame_attributes: Union[bool, float, str],
    allowed_attributes: Union[bool, float, str, List[float], List[str]],
) -> bool:
    """Check if attributes for current frame are allowed.

    Args:
        frame_attributes: Attributes of current frame.
        allowed_attributes: Attributes allowed.

    Returns:
        boolean, whether frame attributes are allowed.
    """
    if isinstance(allowed_attributes, list):
        # assert frame_attributes not in allowed_attributes
        return frame_attributes in allowed_attributes
    return frame_attributes == allowed_attributes


def filter_attributes(
    frames: List[Frame],
    attributes_dict: Optional[
        Dict[str, Union[bool, float, str, List[float], List[str]]]
    ],
) -> List[Frame]:
    """Filter samples according to allowed attributes.

    Args:
        frames: A list of Frame instances to filter.
        attributes_dict: Dictionary of allowed attributes. Each dictionary
            entry contains all allowed attributes for that key.

    Returns:
        A list of filtered Frame instances.

    Raises:
        ValueError: If the filtering removes all frames, we throw an error.
    """
    if attributes_dict:
        for attributes_key in attributes_dict:
            attributes = attributes_dict[attributes_key]
            frames = [
                f
                for f in frames
                if f.attributes
                and check_attributes(f.attributes[attributes_key], attributes)
            ]
        if len(frames) == 0:
            raise ValueError("Dataset empty after filtering by attributes!")
    return frames


# reference:
# https://github.com/facebookresearch/detectron2/blob/7f8f29deae278b75625872c8a0b00b74129446ac/detectron2/data/common.py#L109
class DatasetFromList(torch.utils.data.Dataset):  # type: ignore
    """Wrap a list to a torch Dataset.

    We serialize and wrap big python objects in a torch.Dataset due to a
    memory leak when dealing with large python objects using multiple workers.
    See: https://github.com/pytorch/pytorch/issues/13246
    """

    def __init__(  # type: ignore
        self, lst: List[Any], deepcopy: bool = False, serialize: bool = True
    ):
        """Init.

        Args:
            lst: a list which contains elements to produce.
            deepcopy: whether to deepcopy the element when producing it, s.t.
            the result can be modified in place without affecting the source
            in the list.
            serialize: whether to hold memory using serialized objects. When
            enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
        """
        self._copy = deepcopy
        self._serialize = serialize

        def _serialize(data: Any) -> NDArrayUI8:  # type: ignore
            """Serialize python object to numpy array."""
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            self._lst = [_serialize(x) for x in lst]
            self._addr: NDArrayI64 = np.asarray(
                [len(x) for x in self._lst], dtype=np.int64
            )
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)  # type: ignore
        else:
            self._lst = lst  # pragma: no cover

    def __len__(self) -> int:
        """Return len of list."""
        if self._serialize:
            return len(self._addr)
        return len(self._lst)  # pragma: no cover

    def __getitem__(self, idx: int) -> Any:  # type: ignore
        """Return item of list at idx."""
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes_ = memoryview(self._lst[start_addr:end_addr])  # type: ignore
            return pickle.loads(bytes_)
        if self._copy:  # pragma: no cover
            return copy.deepcopy(self._lst[idx])

        return self._lst[idx]  # pragma: no cover


def all_gather_predictions(
    predictions: Dict[str, List[Frame]],
    pl_module: pl.LightningModule,
    collect_device: str,
) -> Optional[Dict[str, List[Frame]]]:  # pragma: no cover
    """Gather prediction dict in distributed setting."""
    if collect_device == "gpu":
        predictions_list = all_gather_object_gpu(predictions, pl_module)
    elif collect_device == "cpu":
        predictions_list = all_gather_object_cpu(predictions, pl_module)
    else:
        raise ValueError(f"Collect device {collect_device} unknown.")

    if predictions_list is None:
        return None

    result = {}
    for key in predictions:
        prediction_list = [p[key] for p in predictions_list]
        result[key] = list(itertools.chain(*prediction_list))
    return result


def all_gather_gts(
    gts: List[Frame], pl_module: pl.LightningModule, collect_device: str
) -> Optional[List[Frame]]:  # pragma: no cover
    """Gather gts list in distributed setting."""
    if collect_device == "gpu":
        gts_list = all_gather_object_gpu(gts, pl_module)
    elif collect_device == "cpu":
        gts_list = all_gather_object_cpu(gts, pl_module)
    else:
        raise ValueError(f"Collect device {collect_device} unknown.")

    if gts_list is None:
        return None

    return list(itertools.chain(*gts_list))
