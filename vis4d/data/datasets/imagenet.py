"""ImageNet 1k dataset."""

from __future__ import annotations

import os
import tarfile
from collections.abc import Sequence

import numpy as np

from vis4d.data.const import CommonKeys as K
from vis4d.data.typing import DictData

from .base import Dataset
from .util import im_decode, to_onehot


class ImageNet(Dataset):
    """ImageNet 1K dataset."""

    DESCRIPTION = """ImageNet is a large visual database designed for use in
        visual object recognition software research."""
    HOMEPAGE = "http://www.image-net.org/"
    PAPER = "http://www.image-net.org/papers/imagenet_cvpr09.pdf"
    LICENSE = "http://www.image-net.org/terms-of-use"

    KEYS = [K.images, K.categories]

    def __init__(
        self,
        data_root: str,
        keys_to_load: Sequence[str] = (K.images, K.categories),
        split: str = "train",
        num_classes: int = 1000,
    ) -> None:
        """Initialize ImageNet dataset.

        Args:
            data_root (str): Path to root directory of dataset.
            keys_to_load (list[str], optional): List of keys to load. Defaults
                to (K.images, K.categories).
            split (str, optional): Dataset split to load. Defaults to "train".
            num_classes (int, optional): Number of classes to load. Defaults to
                1000.

        NOTE: The dataset is expected to be in the following format:
            data_root
            ├── train
            │   ├── n01440764.tar
            │   ├── ...
            └── val
                ├── n01440764.tar
                ├── ...
            With each tar file containing the images of a single class. The
            images are expected to be in ".JPEG" extension.

            Currently, we are not using the DataBackend for loading the tars to
            avoid keeping too many file pointers open at the same time.
        """
        self.data_root = data_root
        self.keys_to_load = keys_to_load
        self.split = split
        self.num_classes = num_classes
        self.data_infos = []

        self._classes = []
        for file in os.listdir(os.path.join(data_root, split)):
            if file.endswith(".tar"):
                self._classes.append(file)
        assert (
            len(self._classes) == num_classes
        ), f"Expected {num_classes} classes, but found {len(self._classes)}."
        self._classes = sorted(self._classes)

        for class_idx, file in enumerate(self._classes):
            with tarfile.open(os.path.join(data_root, split, file)) as f:
                members = f.getmembers()
                for member in members:
                    if member.isfile() and member.name.endswith(".JPEG"):
                        self.data_infos.append((class_idx, member.name))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        class_idx, image_name = self.data_infos[idx]
        with tarfile.open(
            os.path.join(self.data_root, self.split, self._classes[class_idx])
        ) as f:
            im_bytes = f.extractfile(image_name)
            assert im_bytes is not None, f"Could not extract {image_name}"
            image = im_decode(im_bytes.read())

        data_dict = {}
        if K.images in self.keys_to_load:
            data_dict[K.images] = np.ascontiguousarray(
                image, dtype=np.float32
            )[np.newaxis, ...]
        if K.categories in self.keys_to_load:
            data_dict[K.categories] = to_onehot(
                np.array(class_idx, dtype=np.int64), self.num_classes
            )
        return data_dict
