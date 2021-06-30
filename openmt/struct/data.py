"""Data structure for struct container."""
import abc
import itertools
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from scalabel.eval.mot import EvalResults as MOTEvalResults
from scalabel.label.typing import Frame

NDArrayF64 = npt.NDArray[np.float64]
NDArrayUI8 = npt.NDArray[np.uint8]
TorchCheckpoint = Dict[str, Union[int, str, Dict[str, NDArrayF64]]]
LossesType = Dict[str, torch.Tensor]
EvalResult = Union[Dict[str, float], MOTEvalResults]
EvalResults = Dict[str, Union[Dict[str, float], MOTEvalResults]]


class DataInstance(metaclass=abc.ABCMeta):
    """Meta class for input data."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Len method. Return -1 if not applicable."""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> "DataInstance":
        """Get item method."""
        raise NotImplementedError

    @abc.abstractmethod
    def to(  # pylint: disable=invalid-name
        self, device: torch.device
    ) -> "DataInstance":
        """Move to device (CPU / GPU / ...)."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def cat(cls, instances: List["DataInstance"]) -> "DataInstance":
        """Concatenate two data instances."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Returns current device if applicable."""
        raise NotImplementedError


class Images(DataInstance):
    """Data structure for saving images."""

    stride: int = 32

    def __init__(
        self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]
    ):
        """Init Images class.

        Args:
            tensor (torch.Tensor): shape (N, C_1, ..., C_K, H, W) where K >= 1.
            image_sizes (list[tuple[int, int]]): Each tuple is (w, h). It can
                be smaller than (w, h) due to padding.
        """
        assert len(tensor.shape) > 3
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_sizes)

    def __getitem__(self, idx: int) -> "Images":
        """Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (C_1, ..., C_K, H, W)
            where K >= 1
        """
        size = self.image_sizes[idx]
        return Images(
            self.tensor[idx : idx + 1, ..., : size[1], : size[0]], [size]
        )

    def to(self, device: torch.device) -> "Images":
        """Put images on device."""
        cast_tensor = self.tensor.to(device)
        return Images(cast_tensor, self.image_sizes)

    @classmethod
    def cat(cls, instances: List["Images"]) -> "Images":  # type: ignore
        """Concatenate two Images objects."""
        assert isinstance(instances, (list, tuple))
        assert len(instances) > 0
        assert all((isinstance(inst, Images) for inst in instances))
        max_hw = (
            max([im.tensor.shape[-2] for im in instances]),
            max([im.tensor.shape[-1] for im in instances]),
        )
        lens = [len(x) for x in instances]

        # ensure divisibility by stride
        pad = lambda x: (x + (cls.stride - 1)) // cls.stride * cls.stride
        max_hw = (pad(x) for x in max_hw)  # type: ignore

        batch_shape = (
            [sum(lens)] + list(instances[0].tensor.shape[1:-2]) + list(max_hw)
        )
        pad_imgs = instances[0].tensor.new_full(batch_shape, 0.0)
        cum_len = 0
        for img, cur_len in zip(instances, lens):
            pad_imgs[
                cum_len : cum_len + cur_len,
                ...,
                : img.tensor.shape[-2],
                : img.tensor.shape[-1],
            ].copy_(img.tensor)
            cum_len += cur_len

        all_sizes = list(
            itertools.chain.from_iterable([x.image_sizes for x in instances])
        )
        return Images(pad_imgs.contiguous(), all_sizes)

    @property
    def device(self) -> torch.device:
        """Returns current device."""
        return self.tensor.device


class InputSample:
    """Container holding varying types of DataInstances and Frame metadata."""

    def __init__(
        self, metadata: Frame, image: Images, **kwargs: DataInstance
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.image = image
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __setattr__(self, key: str, val: Union[Frame, DataInstance]) -> None:
        """Set attribute."""
        assert isinstance(val, (DataInstance, Frame))
        super().__setattr__(key, val)

    def __getattr__(self, key: str) -> DataInstance:
        """Get attribute."""
        if key in self.dict():
            return getattr(self, key)  # type: ignore
        raise AttributeError(f"Could not find attribute {key}.")

    def dict(self) -> Dict[str, Union[Frame, DataInstance]]:
        """Return InputData object as dict."""
        return self.__dict__
