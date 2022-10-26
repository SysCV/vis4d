"""Stanford 3D indoor dataset."""
import glob
import os
from io import BytesIO
from multiprocessing.sharedctypes import Value
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from vis4d.common.typing import DictStrAny
from vis4d.data.const import COMMON_KEYS
from vis4d.data.io.base import DataBackend
from vis4d.data.io.file import FileBackend

from .base import Dataset, DictData
from .util import CacheMappingMixin

# s3dis semantic mappings


class S3DIS(Dataset, CacheMappingMixin):
    """S3DIS dataset class."""

    _DESCRIPTION = """S3DIS is a large-scale indoor pointcloud dataset."""
    _TASKS = ["3DSegment"]
    _URL = "https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf"

    CLASS_NAME_TO_IDX = {
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
    CLASS_COUNTS = torch.Tensor(
        [
            37334028,
            32206900,
            52921324,
            4719832,
            4145093,
            4127868,
            10681455,
            7930065,
            6318085,
            9209662,
            21825992,
            949299,
            2457821,
        ]
    )

    AVAILABLE_KEYS: List[str] = [
        COMMON_KEYS.points3d,
        COMMON_KEYS.colors3d,
        COMMON_KEYS.semantics3d,
        COMMON_KEYS.instances3d,
    ]

    COLOR_MAPPING = torch.tensor(
        [
            [152, 223, 138],
            [31, 119, 180],
            [188, 189, 34],
            [140, 86, 75],
            [255, 152, 150],
            [214, 39, 40],
            [197, 176, 213],
            [23, 190, 207],
            [178, 76, 76],
            [247, 182, 210],
            [66, 188, 102],
            [219, 219, 141],
            [140, 57, 197],
            [202, 185, 52],
        ]
    )

    def __init__(
        self,
        data_root: str,
        split: str = "trainNoArea5",
        data_backend: Optional[DataBackend] = None,
        keys_to_load: List[str] = AVAILABLE_KEYS,
    ) -> None:
        """Creates a new S3DIS dataset.
        Args:
        data_root (str): Path to S3DIS folder
        split (str): which split to load. Must either be
            trainNoArea[1-6] or testArea[1-6].
            e.g. trainNoArea5 will load all areas except area 5 and
            testArea5 will only load area 5
        """

        super().__init__()

        self.data_root = data_root
        self.split = split
        self.data_backend = (
            data_backend if data_backend is not None else FileBackend()
        )

        self.areas: List[str] = [
            "Area_1",
            "Area_2",
            "Area_3",
            "Area_4",
            "Area_5",
            "Area_6",
        ]
        area_number = int(self.split.split("Area")[-1])
        if "trainNoArea" in self.split:
            self.areas.remove(self.areas[area_number - 1])
        elif "testArea" in self.split:
            self.areas = [self.areas[area_number - 1]]
        else:
            raise ValueError("Unknown split: ", self.split)

        self.data = self._load_mapping(
            self._generate_data_mapping,
            os.path.join(self.data_root, self.split + ".json"),
        )
        self.keys_to_load = keys_to_load

    @property
    def num_classes(self):
        """The number of classes int he datset."""
        return len(S3DIS.CLASS_NAME_TO_IDX)

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
                        class_label=S3DIS.CLASS_NAME_TO_IDX.get(sem_name, 12),
                        path=anns,
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
        instance_ids = np.zeros((0, 1), dtype=int)

        for values in data.values():
            data_path = values["path"]
            instance_id = int(
                values["path"].split("_")[-1].replace(".txt", "")
            )
            np_data = pd.read_csv(
                BytesIO(self.data_backend.get(data_path)),
                header=None,
                delimiter=" ",
            ).values.astype(np.float32)

            if COMMON_KEYS.points3d in self.keys_to_load:
                coords = np.vstack([coords, np_data[:, :3]])
            if COMMON_KEYS.colors3d in self.keys_to_load:
                color = np.vstack([color, np_data[:, 3:]])
            if COMMON_KEYS.semantics3d in self.keys_to_load:
                semantic_ids = np.vstack(
                    [
                        semantic_ids,
                        np.ones((np_data.shape[0], 1), dtype=int)
                        * values["class_label"],
                    ]
                )
            if COMMON_KEYS.instances3d in self.keys_to_load:
                instance_ids = np.vstack(
                    [
                        instance_ids,
                        np.ones((np_data.shape[0], 1), dtype=int)
                        * instance_id,
                    ]
                )
        coords -= np.amin(coords, axis=0)

        data = dict()
        for key in self.keys_to_load:
            if key == COMMON_KEYS.points3d:
                data[key] = torch.from_numpy(coords)
            elif key == COMMON_KEYS.colors3d:
                data[key] = torch.from_numpy(color / 255)
            elif key == COMMON_KEYS.semantics3d:
                data[key] = torch.from_numpy(semantic_ids).squeeze(-1)
            elif key == COMMON_KEYS.instances3d:
                data[key] = torch.from_numpy(instance_ids).squeeze(-1)
            else:
                raise ValueError(f"Can not load data for key: {key}")
        return data
