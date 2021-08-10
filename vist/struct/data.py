"""OpenMT Input data structures."""

import itertools
from typing import Dict, List, Optional, Tuple, Union

import torch
from scalabel.label.typing import Frame

from .labels import Boxes2D, Boxes3D
from .structures import DataInstance


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
    def cat(
        cls, instances: List["Images"], device: Optional[torch.device] = None
    ) -> "Images":
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
        if device is None:
            device = instances[0].tensor.device
        pad_imgs = torch.zeros(batch_shape, device=device)
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
        self,
        metadata: Frame,
        image: Images,
        boxes2d: Optional[Boxes2D] = None,
        boxes3d: Optional[Boxes3D] = None,
        **kwargs: DataInstance,
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.image = image
        if boxes2d is not None:
            self.boxes2d = boxes2d
        else:
            self.boxes2d = Boxes2D(torch.empty(0, 5))
        if boxes3d is not None:
            self.boxes3d = boxes3d
        else:
            self.boxes3d = Boxes3D(torch.empty(0, 8))

        self.attributes = dict()  # type: Dict[str, DataInstance]
        for k, v in kwargs.items():
            self.attributes[k] = v

    def get(self, key: str) -> Union[Frame, DataInstance]:
        """Get attribute by key."""
        if key == "metadata":
            return self.metadata
        if key == "image":
            return self.image
        if key == "boxes2d":
            return self.boxes2d
        if key == "boxes3d":
            return self.boxes3d
        if key in self.attributes:
            return self.attributes[key]
        raise AttributeError(f"Attribute {key} not found!")

    def dict(self) -> Dict[str, Union[Frame, DataInstance]]:
        """Return InputSample object as dict."""
        obj_dict = {
            "metadata": self.metadata,
            "image": self.image,
            "boxes2d": self.boxes2d,
            "boxes3d": self.boxes3d,
        }  # type: Dict[str, Union[Frame, DataInstance]]
        obj_dict.update(self.attributes)
        return obj_dict
