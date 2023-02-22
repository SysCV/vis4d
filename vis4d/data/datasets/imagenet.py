"""ImageNet 1k dataset."""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np

from vis4d.common import DictStrAny
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.io.base import DataBackend
from vis4d.data.io.file import FileBackend
from vis4d.data.typing import DictData

from .base import Dataset
from .util import im_decode


class ImageNet(Dataset):
    """ImageNet 1k dataset."""

    DESCRIPTION = "ImageNet 1k dataset."
    HOMEPAGE = "http://www.image-net.org/"
    LICENSE = "http://www.image-net.org/terms-of-use"

    _KEYS = [Keys.images, Keys.categories]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        keys_to_load: tuple[str, ...] = (Keys.images, Keys.categories),
        backend: DataBackend = FileBackend(),
    ):
        """Initialize ImageNet 1k dataset.

        Args:
            data_root (str): Root directory of dataset.
            split (str, optional): Dataset split. Defaults to "train".
            keys_to_load (tuple[str, ...], optional): Keys to load. Defaults to
                (Keys.images, Keys.categories).
            backend (DataBackend, optional): Data backend. Defaults to
                FileBackend().
        """
        self.keys_to_load = keys_to_load
        self.split = split
        self.data_root = data_root
        self.backend = backend
        self._data_infos = self._load_data()
        self._len = len(self._data_infos)

    def __len__(self) -> int:
        """Get length of dataset."""
        return self._len

    def _load_img(self, filepath: str) -> np.ndarray:
        """Load image from path."""
        return im_decode(self.backend.read(filepath))

    def _load_data(self) -> list[DictStrAny]:
        """Load data infos from imagenet annotation file."""
        with contextlib.redirect_stdout(io.StringIO()):
            data_infos = [
                {
                    "image_path": os.path.join(
                        self.data_root, self.split, line.split()[0]
                    ),
                    "label": int(line.split()[1]),
                }
                for line in self.backend.read(
                    os.path.join(self.data_root, f"{self.split}.txt")
                ).splitlines()
            ]
        return data_infos

    def __getitem__(self, idx: int) -> DictData:
        """Get item from dataset."""
        data = self._data_infos[idx]
        dict_data = {}
        if Keys.images in self.keys_to_load:
            dict_data[Keys.images] = self._load_img(data["image_path"])
        if Keys.categories in self.keys_to_load:
            dict_data[Keys.categories] = data["label"]
        return dict_data
