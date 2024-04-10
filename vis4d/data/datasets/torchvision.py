"""Provides functionalities to wrap torchvision datasets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PIL.Image import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from ..const import CommonKeys as K
from ..typing import DictData
from .base import Dataset


class TorchvisionDataset(Dataset):
    """Wrapper for torchvision datasets.

    This class wraps torchvision datasets and converts them to the format that
    is expected by the vis4d framework.

    The return of the torchvisons dataset is passed to the data_converter,
    which needs to be provided by the user. The data_converter is expected to
    return a DictData object following the vis4d conventions.

    For well defined dataformats, such as classification, there
    are already implemented wrappers that can be used. See
    `TorchvisionClassificationDataset` for an example.
    """

    def __init__(  # type: ignore
        self,
        torchvision_ds: VisionDataset,
        data_converter: Callable[[Any], DictData],
    ) -> None:
        """Creates a new instance of the class.

        Args:
            torchvision_ds (VisionDataset): Torchvision dataset that should be
                converted.
            data_converter (Callable[[Any], DictData]): Function that
                converts the output of the torchvision datasets __getitem__
                to the format expected by the vis4d framework.
        """
        super().__init__()
        self.torchvision_ds = torchvision_ds
        self.data_converter = data_converter

    def __getitem__(self, idx: int) -> DictData:
        """Returns a new sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            DictData: Data in vis4d format.
        """
        return self.data_converter(self.torchvision_ds[idx])

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.torchvision_ds)


class TorchvisionClassificationDataset(TorchvisionDataset):
    """Wrapper for torchvision classification datasets.

    This class wraps torchvision classification datasets and converts them to
    the format that is expected by the vis4d framework.

    It expects the torchvision dataset to return a tuple of (image, class_id)
    where the image is a PIL Image and the class_id is an integer.

    If you want to use a torchvision dataset that returns a different format,
    you can provide a custom data_converter function to the
    `TorchvisionDataset` class.

    The returned sample will have the following key, values:
    images: ndarray of dimension (1, H, W, C)
    categories: ndarray of dimension 1.

    Example:
    >>> from torchvision.datasets.mnist import MNIST
    >>> ds = TorchvisionClassificationDataset(
    >>>     MNIST("data/mnist_ds", train=False)
    >>> )
    >>> data = next(iter(ds))
    >>> print(data.keys)
    dict_keys(['images', 'categories'])
    """

    def __init__(self, detection_ds: VisionDataset) -> None:
        """Creates a new instance of the class.

        Args:
            detection_ds (VisionDataset): Torchvision dataset that
                returns a tuple of (image, class_id) where the image is a PIL
                Image and the class_id is an integer.
        """
        img_to_tensor = ToTensor()

        def _data_converter(img_and_target: tuple[Image, int]) -> DictData:
            """Converts the output of a torchvision dataset.

            The output is converted to the format expected by the vis4d
            framework.

            Args:
                img_and_target (tuple[Image, int]): Output of the datasets
                    __getitem__ method.

            Returns:
                DictData: Sample in vis4d format.
            """
            img, class_id = img_and_target
            data: DictData = {}
            data[K.images] = (
                img_to_tensor(img).unsqueeze(0).permute(0, 2, 3, 1).numpy()
            )
            data[K.categories] = np.array([class_id], dtype=np.int64)

            return data

        super().__init__(detection_ds, _data_converter)
