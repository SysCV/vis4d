"""ImageNet 1k dataset."""

from __future__ import annotations

import os
from collections.abc import Sequence

import h5py
import numpy as np
import torch

from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.typing import DictData

from .base import Dataset
from .util import im_decode


class ImageNet(Dataset):
    """ImageNet 1K dataset."""

    DESCRIPTION = """ImageNet is a large visual database designed for use in
        visual object recognition software research."""
    HOMEPAGE = "http://www.image-net.org/"
    PAPER = "http://www.image-net.org/papers/imagenet_cvpr09.pdf"
    LICENSE = "http://www.image-net.org/terms-of-use"

    KEYS = [Keys.images, Keys.categories]

    def __init__(
        self,
        data_root: str,
        keys_to_load: Sequence[str] = (Keys.images, Keys.categories),
        split: str = "train",
        num_classes: int = 1000,
    ) -> None:
        """Initialize ImageNet dataset.

        Args:
            data_root (str): Path to root directory of dataset.
            keys_to_load (list[str], optional): List of keys to load. Defaults
                to (Keys.images, Keys.categories).
            split (str, optional): Dataset split to load. Defaults to "train".
            num_classes (int, optional): Number of classes to load. Defaults to
                1000.
        NOTE: The dataset is expected to be in the following format:
            data_root
            ├── train.hdf5
            ├── val.hdf5
            With each h5 file containing the images in the following format:
            `class_idx/img_name.JPEG`
        """
        self.data_root = data_root
        self.keys_to_load = keys_to_load
        self.split = split
        self.num_classes = num_classes

        self.file = h5py.File(os.path.join(data_root, f"{split}.hdf5"), "r")
        self.groups = list(self.file.keys())
        self.group_sizes = [len(self.file[group]) for group in self.groups]

    def __len__(self) -> int:
        """Return length of dataset."""
        return sum(self.group_sizes)

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        group_idx = 0
        while idx >= self.group_sizes[group_idx]:
            idx -= self.group_sizes[group_idx]
            group_idx += 1
        group_name = self.groups[group_idx]
        img_name = list(self.file[group_name].keys())[idx]
        img_bytes = bytearray(self.file[group_name][img_name])
        img = im_decode(img_bytes)

        data_dict = {}
        if Keys.images in self.keys_to_load:
            data_dict[Keys.images] = torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1)),
                dtype=torch.float32,
            ).unsqueeze(0)
        if Keys.categories in self.keys_to_load:
            data_dict[Keys.categories] = torch.tensor(
                group_idx, dtype=torch.long
            ).unsqueeze(0)
        return data_dict
