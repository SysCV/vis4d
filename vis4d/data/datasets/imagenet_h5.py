"""ImageNet 1k dataset."""

from __future__ import annotations

import os
import h5py
import pickle
import tarfile
from collections.abc import Sequence

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
        self._load_data_infos()

    def _load_data_infos(self) -> None:
        """Load data infos from disk."""
        timer = Timer()
        sample_list_path = os.path.join(self.data_root, f"{self.split}.pkl")
        with open(sample_list_path, "rb") as f:
            sample_list = pickle.load(f)[0]
        if sample_list[-1][1] != self.num_classes - 1:
            raise ValueError("Number of classes unmatched!")
        self.data_infos = [(member.name, class_idx) for member, class_idx in sample_list]
        rank_zero_info(f"Loading {self} takes {timer.time():.2f} seconds.")

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        timer = Timer()
        img_name, class_idx = self.data_infos[idx]
        group_name = img_name.split('_')[0]
        img_bytes = bytearray(self.file[f"{group_name}/{img_name}"])
        img = im_decode(img_bytes)

        data_dict = {}
        if Keys.images in self.keys_to_load:
            data_dict[Keys.images] = torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1)),
                dtype=torch.float32,
            ).unsqueeze(0)
        if Keys.categories in self.keys_to_load:
            data_dict[Keys.categories] = torch.tensor(
                class_idx, dtype=torch.long
            ).unsqueeze(0)
        t = timer.time()
        if t > 1.0:
            rank_zero_info(f"idx: {idx} time: {t:.3f} name: {img_name}")
        return data_dict
