"""COCO dataset."""
import contextlib
import io
import os
from typing import List

import numpy as np
import torch
from pycocotools.coco import COCO as COCOAPI

from vis4d.data.io.file import FileBackend
from vis4d.data_to_revise.utils import im_decode
from vis4d.struct_to_revise import DictStrAny

from .base import BaseDataset, DataKeys, DictData
from .utils import CacheMappingMixin


class COCO(BaseDataset, CacheMappingMixin):
    """COCO dataset class."""

    def __init__(self, data_root: str, split: str = "train2017") -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.data_backend = FileBackend()

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

        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        data = []
        for img_id, img in zip(img_ids, imgs):
            data.append(
                dict(img_id=img_id, img=img, anns=coco_api.imgToAnns[img_id])
            )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> DictData:
        """Transform coco sample to vis4d input format.

        Returns:
            image
            boxes2d
        """
        data = self.data[idx]
        img_path = os.path.join(
            self.data_root, self.split, data["img"]["file_name"]
        )
        im_bytes = self.data_backend.get(img_path)
        img = im_decode(im_bytes)
        img = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1)),
            dtype=torch.float32,
        ).unsqueeze(0)
        boxes = []
        for ann in data["anns"]:
            x1, y1, width, height = ann["bbox"]
            x2, y2 = x1 + width, y1 + height
            boxes.append((x1, y1, x2, y2))
        return {
            DataKeys.images: img,
            DataKeys.boxes2d: torch.tensor(boxes, dtype=torch.float32),
        }
