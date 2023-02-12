"""COCO dataset."""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pycocotools.mask as maskUtils
import torch
from pycocotools.coco import COCO as COCOAPI

from vis4d.common import DictStrAny
from vis4d.data.const import CommonKeys
from vis4d.data.io.base import DataBackend
from vis4d.data.io.file import FileBackend
from vis4d.data.typing import DictData

from .base import Dataset
from .util import CacheMappingMixin, im_decode

# COCO detection
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
coco_seg_map = {
    "background": 0,
    "airplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "dining table": 11,
    "dog": 12,
    "horse": 13,
    "motorcycle": 14,
    "person": 15,
    "potted plant": 16,
    "sheep": 17,
    "couch": 18,
    "train": 19,
    "tv": 20,
}


class COCO(Dataset, CacheMappingMixin):
    """COCO dataset class."""

    DESCRIPTION = """COCO is a large-scale object detection, segmentation, and
    captioning dataset."""
    URL = "http://cocodataset.org/#home"
    KEYS = ["images", "boxes2d", "boxes2d_classes", "masks"]

    def __init__(
        self,
        data_root: str,
        keys_to_load: tuple[str, ...] = (
            CommonKeys.images,
            CommonKeys.boxes2d,
            CommonKeys.boxes2d_classes,
            CommonKeys.masks,
        ),
        split: str = "train2017",
        remove_empty: bool = False,
        minimum_box_area: float = 0,
        use_pascal_voc_cats: bool = False,
        data_backend: None | DataBackend = None,
    ) -> None:
        """Initialize the COCO dataset.

        Args:
            data_root (str): Path to the root directory of the dataset.
            keys_to_load (tuple[str, ...]): Keys to load from the dataset.
            split (split): Which split to load. Default: "train2017".
            remove_empty (bool): Whether to remove images with no annotations.
            minimum_box_area (float): Minimum area of the bounding boxes.
                Default: 0.
            use_pascal_voc_cats (bool): Whether to use Pascal VOC categories.
            data_backend (None | DataBackend): Data backend to use.
                Default: None.
        """
        super().__init__()

        self.data_root = data_root
        self.keys_to_load = keys_to_load
        self.split = split
        self.remove_empty = remove_empty
        self.minimum_box_area = minimum_box_area
        self.use_pascal_voc_cats = use_pascal_voc_cats
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )

        # handling keys to load
        self.validate_keys(keys_to_load)
        self.with_images = CommonKeys.images in keys_to_load
        self.with_boxes = (CommonKeys.boxes2d in keys_to_load) or (
            CommonKeys.boxes2d_classes in keys_to_load
        )
        self.with_masks = CommonKeys.masks in keys_to_load

        self.data = self._load_mapping(self._generate_data_mapping)

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return (
            f"COCODataset(root={self.data_root}, split={self.split}, "
            f"use_pascal_voc_cats={self.use_pascal_voc_cats})"
        )

    def _has_valid_annotation(self, anns: list[dict[str, float]]) -> bool:
        """Filter empty or low occupied samples."""
        if self.remove_empty and len(anns) == 0:
            return False
        return sum(ann["area"] for ann in anns) >= self.minimum_box_area

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generate coco dataset mapping."""
        annotation_file = os.path.join(
            self.data_root, "annotations", "instances_" + self.split + ".json"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCOAPI(annotation_file)
        cat_ids = sorted(coco_api.getCatIds())
        cats_map = {c["id"]: c["name"] for c in coco_api.loadCats(cat_ids)}
        if self.use_pascal_voc_cats:
            voc_cats = set(coco_seg_map.keys())

        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        samples = []
        for img_id, img in zip(img_ids, imgs):
            anns = coco_api.imgToAnns[img_id]
            if self.use_pascal_voc_cats:
                anns = [
                    ann
                    for ann in anns
                    if cats_map[ann["category_id"]] in voc_cats
                ]
            for ann in anns:
                cat_name = cats_map[ann["category_id"]]
                if self.use_pascal_voc_cats:
                    ann["category_id"] = coco_seg_map[cat_name]
                else:
                    ann["category_id"] = coco_det_map[cat_name]
            if self._has_valid_annotation(anns):
                samples.append(dict(img_id=img_id, img=img, anns=anns))
        return samples

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> DictData:
        """Transform coco sample to vis4d input format.

        Returns:
            DataDict[DataKeys, Union[torch.Tensor, Dict[Any]]]
        """
        data = self.data[idx]
        img_h, img_w = data["img"]["height"], data["img"]["width"]
        dict_data = {
            CommonKeys.original_hw: [img_h, img_w],
            CommonKeys.input_hw: [img_h, img_w],
            "coco_image_id": data["img"]["id"],
        }

        if self.with_images:
            img_path = os.path.join(
                self.data_root, self.split, data["img"]["file_name"]
            )
            im_bytes = self.data_backend.get(img_path)
            img = im_decode(im_bytes)
            img_tensor = torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1)),
                dtype=torch.float32,
            ).unsqueeze(0)
            assert (img_h, img_w) == img_tensor.shape[
                2:
            ], "Image's shape doesn't match annotation."
            dict_data[CommonKeys.images] = img_tensor

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
            if not boxes:
                box_tensor = torch.empty((0, 4), dtype=torch.float32)
                mask_tensor = torch.empty((0, img_h, img_w), dtype=torch.uint8)
            else:
                box_tensor = torch.tensor(boxes, dtype=torch.float32)
                mask_tensor = torch.as_tensor(
                    np.ascontiguousarray(masks), dtype=torch.uint8
                )

            if CommonKeys.boxes2d in self.keys:
                dict_data[CommonKeys.boxes2d] = box_tensor
            if CommonKeys.boxes2d_classes in self.keys:
                dict_data[CommonKeys.boxes2d_classes] = torch.tensor(
                    classes, dtype=torch.long
                )
            if self.with_masks:
                dict_data[CommonKeys.masks] = mask_tensor

        return dict_data
