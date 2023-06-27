"""ImageNet 1k dataset."""

from __future__ import annotations

import os
import pickle
import tarfile
from collections.abc import Sequence

import numpy as np

from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.common.typing import ArgsType
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
        use_sample_lists: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize ImageNet dataset.

        Args:
            data_root (str): Path to root directory of dataset.
            keys_to_load (list[str], optional): List of keys to load. Defaults
                to (K.images, K.categories).
            split (str, optional): Dataset split to load. Defaults to "train".
            num_classes (int, optional): Number of classes to load. Defaults to
                1000.
            use_sample_lists (bool, optional): Whether to use sample lists for
                loading the dataset. Defaults to False.

        NOTE: The dataset is expected to be in the following format:
            data_root
            ├── train.pkl  # Sample lists for training set (optional)
            ├── val.pkl    # Sample lists for validation set (optional)
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
        super().__init__(**kwargs)
        self.data_root = data_root
        self.keys_to_load = keys_to_load
        self.split = split
        self.num_classes = num_classes
        self.use_sample_lists = use_sample_lists
        self.data_infos: list[tuple[tarfile.TarInfo, int]] = []
        self._classes: list[str] = []
        self._load_data_infos()

    def _load_data_infos(self) -> None:
        """Load data infos from disk."""
        timer = Timer()
        # Load tar files
        for file in os.listdir(os.path.join(self.data_root, self.split)):
            if file.endswith(".tar"):
                self._classes.append(file)
        assert len(self._classes) == self.num_classes, (
            f"Expected {self.num_classes} classes, but found "
            f"{len(self._classes)} tar files."
        )
        self._classes = sorted(self._classes)

        sample_list_path = os.path.join(self.data_root, f"{self.split}.pkl")
        if self.use_sample_lists and os.path.exists(sample_list_path):
            with open(sample_list_path, "rb") as f:
                sample_list = pickle.load(f)[0]
                if sample_list[-1][1] == self.num_classes - 1:
                    self.data_infos = sample_list
                else:
                    raise ValueError(
                        "Sample list does not match the number of classes. "
                        "Please regenerate the sample list or set "
                        "use_sample_lists=False."
                    )
        # If sample lists are not available, generate them on the fly.
        else:
            for class_idx, file in enumerate(self._classes):
                with tarfile.open(
                    os.path.join(self.data_root, self.split, file)
                ) as f:
                    members = f.getmembers()
                    for member in members:
                        if member.isfile() and member.name.endswith(".JPEG"):
                            self.data_infos.append((member, class_idx))

        rank_zero_info(f"Loading {self} takes {timer.time():.2f} seconds.")

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> DictData:
        """Convert single element at given index into Vis4D data format."""
        member, class_idx = self.data_infos[idx]
        with tarfile.open(
            os.path.join(self.data_root, self.split, self._classes[class_idx]),
            mode="r:*",  # unexclusive read mode
        ) as f:
            im_bytes = f.extractfile(member)
            assert im_bytes is not None, f"Could not extract {member.name}!"
            image = im_decode(im_bytes.read())

        data_dict: DictData = {}
        if K.images in self.keys_to_load:
            data_dict[K.images] = np.ascontiguousarray(
                image, dtype=np.float32
            )[np.newaxis, ...]
            image_hw = image.shape[:2]
            data_dict[K.input_hw] = image_hw
            data_dict[K.original_hw] = image_hw
        if K.categories in self.keys_to_load:
            data_dict[K.categories] = to_onehot(
                np.array(class_idx, dtype=np.int64), self.num_classes
            )
        return data_dict
