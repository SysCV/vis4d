"""Stanford 3D indoor dataset."""

from __future__ import annotations

import copy
import glob
import os
from collections.abc import Sequence
from io import BytesIO

import numpy as np
import pandas as pd
import torch

from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import CommonKeys as K
from vis4d.data.typing import DictData

from .base import Dataset
from .util import CacheMappingMixin


class S3DIS(CacheMappingMixin, Dataset):
    """S3DIS dataset class."""

    DESCRIPTION = """S3DIS is a large-scale indoor pointcloud dataset."""
    HOMEPAGE = "https://buildingparser.stanford.edu/dataset.html"
    PAPER = (
        "https://openaccess.thecvf.com/content_cvpr_2016/papers/"
        "Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf"
    )
    LICENSE = "CC BY-NC-SA 4.0"

    KEYS = [
        K.points3d,
        K.colors3d,
        K.semantics3d,
        K.instances3d,
    ]

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
            3370714,
            2856755,
            4919229,
            318158,
            375640,
            478001,
            974733,
            650464,
            791496,
            88727,
            1284130,
            229758,
            2272837,
        ]
    )

    AVAILABLE_KEYS: Sequence[str] = (
        K.points3d,
        K.colors3d,
        K.semantics3d,
        K.instances3d,
    )

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
        keys_to_load: Sequence[str] = (
            K.points3d,
            K.colors3d,
            K.semantics3d,
            K.instances3d,
        ),
        cache_points: bool = True,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates a new S3DIS dataset.

        Args:
            data_root (str): Path to S3DIS folder
            split (str): which split to load. Must either be trainNoArea[1-6]
                or testArea[1-6].  e.g. trainNoArea5 will load all areas except
                area 5 and testArea5 will only load area 5.
            keys_to_load (list[str]): What kind of data should be loaded
                (e.g. colors, xyz, semantics, ...)
            cache_points (bool): If true caches loaded points instead of
                reading them from the disk every time.
            cache_as_binary (bool): Whether to cache the dataset as binary.
                Default: False.
            cached_file_path (str | None): Path to a cached file. If cached
                file exist then it will load it instead of generating the data
                mapping. Default: None.

        Raises:
            ValueError: If requested split is malformed.
        """
        super().__init__(**kwargs)

        self.data_root = data_root
        self.split = split

        self.areas: list[str] = [
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

        self.data, _ = self._load_mapping(
            self._generate_data_mapping,
            cache_as_binary=cache_as_binary,
            cached_file_path=cached_file_path,
        )
        self.keys_to_load = keys_to_load

        # Cache
        self.cache_points = cache_points
        self._cache: dict[int, DictData] = {}

    @property
    def num_classes(self) -> int:
        """The number of classes int he datset."""
        return len(S3DIS.CLASS_NAME_TO_IDX)

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"S3DIS(root={self.data_root}, split={self.split})"

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generate 3dis dataset mapping."""
        data: list[DictStrAny] = []
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
                    room_data[instance_id] = {
                        "class_label": S3DIS.CLASS_NAME_TO_IDX.get(
                            sem_name, 12
                        ),
                        "path": anns,
                    }
                data.append(room_data)

        return data

    def __len__(self) -> int:
        """Length of the datset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> DictData:
        """Transform s3dis sample to vis4d input format.

        Returns:
            coordinates: 3D Poitns coordinate Shape(n x 3)
            colors: 3D Point colors Shape(n x 3)
            Semantic Classes: 3D Point classes Shape(n x 1)

        Raises:
            ValueError: If a requested key does not exist in this dataset.
        """
        data = self.data[idx]

        # Cache data
        if self.cache_points and idx in self._cache:
            return copy.deepcopy(self._cache[idx])

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

            if K.points3d in self.keys_to_load:
                coords = np.vstack([coords, np_data[:, :3]])
            if K.colors3d in self.keys_to_load:
                color = np.vstack([color, np_data[:, 3:]])
            if K.semantics3d in self.keys_to_load:
                semantic_ids = np.vstack(
                    [
                        semantic_ids,
                        np.ones((np_data.shape[0], 1), dtype=int)
                        * values["class_label"],
                    ]
                )
            if K.instances3d in self.keys_to_load:
                instance_ids = np.vstack(
                    [
                        instance_ids,
                        np.ones((np_data.shape[0], 1), dtype=int)
                        * instance_id,
                    ]
                )

        data = {}
        for key in self.keys_to_load:
            if key == K.points3d:
                data[key] = coords
            elif key == K.colors3d:
                data[key] = color / 255.0
            elif key == K.semantics3d:
                data[key] = semantic_ids.squeeze(-1)
            elif key == K.instances3d:
                data[key] = instance_ids.squeeze(-1)
            else:
                raise ValueError(f"Can not load data for key: {key}")

        if self.cache_points:
            self._cache[idx] = copy.deepcopy(data)
        return data
