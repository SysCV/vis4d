"""COCO dataset."""
import contextlib
import io
import os
from typing import List, Optional

import numpy as np
import pycocotools.mask as maskUtils
import torch
from pycocotools.coco import COCO as COCOAPI

from vis4d.data.io.base import DataBackend
from vis4d.data.io.file import FileBackend
from vis4d.struct_to_revise import DictStrAny

from .base import DataKeys, Dataset, DictData, MultitaskMixin
from .utils import CacheMappingMixin, im_decode

# COCO
coco_det_map = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}

# COCO segmentation categories
coco_seg_cats = [
    0,
    5,
    2,
    16,
    9,
    44,
    6,
    3,
    17,
    62,
    21,
    67,
    18,
    19,
    4,
    1,
    64,
    20,
    63,
    7,
    72,
]


class COCO(Dataset, MultitaskMixin, CacheMappingMixin):
    """COCO dataset class."""

    _DESCRIPTION = """COCO is a large-scale object detection, segmentation, and
    captioning dataset."""
    _KEYS = ["images", "boxes2d", "boxes2d_classes", "masks"]
    _URL = "http://cocodataset.org/#home"

    def __init__(
        self,
        data_root: str,
        keys_to_load: List[str] = [
            DataKeys.images,
            DataKeys.boxes2d,
            DataKeys.boxes2d_classes,
            DataKeys.masks,
        ],
        split: str = "train2017",
        data_backend: Optional[DataBackend] = None,
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.keys_to_load = keys_to_load
        self.split = split
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )

        # handling keys to load
        self.validate_keys(keys_to_load)
        self.with_images = DataKeys.images in keys_to_load
        self.with_boxes = (DataKeys.boxes2d in keys_to_load) or (
            DataKeys.boxes2d_classes in keys_to_load
        )
        self.with_masks = DataKeys.masks in keys_to_load

        self.data = self._load_mapping(
            self._generate_data_mapping,
            os.path.join(
                self.data_root,
                "annotations",
                "instances_" + self.split + ".pkl",
            ),
        )

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"COCODataset(root={self.data_root}, split={self.split})"

    def _generate_data_mapping(self) -> List[DictStrAny]:
        """Generate coco dataset mapping."""
        annotation_file = os.path.join(
            self.data_root, "annotations", "instances_" + self.split + ".json"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCOAPI(annotation_file)

        cat_ids = sorted(coco_api.getCatIds())
        cats_map = {c["id"]: c["name"] for c in coco_api.loadCats(cat_ids)}

        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        data = []
        for img_id, img in zip(img_ids, imgs):
            anns = coco_api.imgToAnns[img_id]
            for ann in anns:
                ann["category_id"] = coco_det_map[cats_map[ann["category_id"]]]
            data.append(dict(img_id=img_id, img=img, anns=anns))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> DictData:
        """Transform coco sample to vis4d input format.

        Returns:
            DataDict[DataKeys, Union[torch.Tensor, Dict[Any]]]
        """
        data = self.data[idx]
        img_h, img_w = data["img"]["height"], data["img"]["width"]
        dict_data = {
            DataKeys.metadata: {
                "original_hw": [img_h, img_w],
                "input_hw": None,
                "transform_params": {},
                "batch_transform_params": {},
                "coco_image_id": data["img"]["id"],
            },
        }

        if self.with_images:
            img_path = os.path.join(
                self.data_root, self.split, data["img"]["file_name"]
            )
            im_bytes = self.data_backend.get(img_path)
            img = im_decode(im_bytes)
            img = torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1)),
                dtype=torch.float32,
            ).unsqueeze(0)
            assert (img_h, img_w) == img.shape[
                2:
            ], "Image's shape doesn't match annotation."
            dict_data[DataKeys.images] = img

        if self.with_boxes or self.with_masks:
            boxes = []
            classes = []
            masks = []
            for ann in data["anns"]:
                x1, y1, width, height = ann["bbox"]
                x2, y2 = x1 + width, y1 + height
                boxes.append((x1, y1, x2, y2))
                classes.append(ann["category_id"])
                mask_ann = ann.get("segmentation", None)
                if mask_ann is not None and self.with_masks:
                    if isinstance(mask_ann, list):
                        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
                        rle = maskUtils.merge(rles)
                    elif isinstance(mask_ann["counts"], list):
                        # uncompressed RLE
                        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
                    else:
                        # RLE
                        rle = mask_ann
                    masks.append(maskUtils.decode(rle))
                else:
                    masks.append(np.empty((img_h, img_w)))
            if not len(boxes):
                box_tensor = torch.empty((0, 4), dtype=torch.float32)
                mask_tensor = torch.empty((0, img_h, img_w), dtype=torch.uint8)
            else:
                box_tensor = torch.tensor(boxes, dtype=torch.float32)
                mask_tensor = torch.as_tensor(
                    np.ascontiguousarray(masks), dtype=torch.uint8
                )

            if DataKeys.boxes2d in self.keys_to_load:
                dict_data[DataKeys.boxes2d] = box_tensor
            if DataKeys.boxes2d_classes in self.keys_to_load:
                dict_data[DataKeys.boxes2d_classes] = torch.tensor(
                    classes, dtype=torch.long
                )
            if self.with_masks:
                dict_data[DataKeys.masks] = mask_tensor

        return dict_data
