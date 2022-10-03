"""Stanford 3D indoor dataset."""
import glob
import os
from io import BytesIO
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from vis4d.data.io.base import BaseDataBackend
from vis4d.data.io.file import FileBackend
from vis4d.struct_to_revise import DictStrAny

from .base import DataKeys, Dataset, DictData
from .utils import CacheMappingMixin

# s3dis semantic mappings
S3DIS_LABELS = {
    "ceiling": 0,
    "floor": 1,
    "wall": 2,
    "beam": 3,
    "column": 4,
    "window": 5,
    "door": 6,
    "chair": 7,
    "table": 8,
    "bookcase": 9,
    "sofa": 10,
    "board": 11,
    "clutter": 12,
}


class S3DIS(Dataset, CacheMappingMixin):
    """S3DIS dataset class."""

    _DESCRIPTION = """S3DIS is a large-scale indoor pointcloud dataset."""
    _TASKS = ["3DSegment"]
    _URL = "https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf"

    def __init__(
        self,
        data_root: str,
        split: str = "trainNoArea5",
        data_backend: Optional[BaseDataBackend] = None,
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )

        self.areas: List[str] = []
        if self.split == "trainNoArea5":
            self.areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]
        elif self.split == "testArea5":
            self.areas = ["Area_5"]
        else:
            raise ValueError("Unknown split: ", self.split)

        self.data = self._load_mapping(
            self._generate_data_mapping,
            os.path.join(self.data_root, self.split + ".json"),
        )

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"S3DIS(root={self.data_root}, split={self.split})"

    def _generate_data_mapping(self) -> List[DictStrAny]:
        """Generate 3dis dataset mapping."""
        data: List[DictStrAny] = []
        for area in self.areas:
            for room_path in glob.glob(
                os.path.join(self.data_root, area + "/*")
            ):
                room_data: DictStrAny = {}
                if not os.path.isdir(room_path):
                    continue

                for anns in glob.glob(
                    os.path.join(room_path, "Annotations/*.txt")
                ):
                    instance_id = os.path.basename(anns.replace(".txt", ""))
                    sem_name = instance_id.split("_")[0]
                    room_data[instance_id] = dict(
                        class_label=S3DIS_LABELS.get(sem_name, 12), path=anns
                    )
                data.append(room_data)

        return data

    def __len__(self) -> int:
        """length of the datset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> DictData:
        """Transform s3dis sample to vis4d input format.

        Returns:
            coordinates: 3D Poitns coordinate Shape(n x 3)
            colors: 3D Point colors Shape(n x 3)
            Semantic Classes: 3D Point classes Shape(n x 1)
        """
        data = self.data[idx]

        coords = np.zeros((0, 3), dtype=np.float32)
        color = np.zeros((0, 3), dtype=np.float32)
        semantic_ids = np.zeros((0, 1), dtype=int)

        for values in data.values():
            data_path = values["path"]
            np_data = pd.read_csv(
                BytesIO(self.data_backend.get(data_path)),
                header=None,
                delimiter=" ",
            ).values.astype(np.float32)

            coords = np.vstack([coords, np_data[:, :3]])
            color = np.vstack([color, np_data[:, 3:]])
            semantic_ids = np.vstack(
                [
                    semantic_ids,
                    np.ones((np_data.shape[0], 1), dtype=int)
                    * values["class_label"],
                ]
            )

        return {
            DataKeys.colors3d: torch.from_numpy(color / 255),
            DataKeys.points3d: torch.from_numpy(coords),
            DataKeys.semantics3d: torch.from_numpy(semantic_ids),
        }
