"""VisT Input data structures."""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from scalabel.label.typing import Frame

from .labels import Boxes2D, Boxes3D, Poly2D
from .structures import DataInstance


class Intrinsics(DataInstance):
    """Data structure for intrinsic calibration."""

    def __init__(self, tensor: torch.Tensor):
        """Init Intrinsics class.

        Args:
            tensor (torch.Tensor): (N, 3, 3) or (3, 3)
        """
        assert 2 <= len(tensor.shape) <= 3
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        self.tensor = tensor

    def to(self, device: torch.device) -> "Intrinsics":
        """Put images on device."""
        cast_tensor = self.tensor.to(device)
        return Intrinsics(cast_tensor)

    def __getitem__(self, item: int) -> "Intrinsics":
        """Return single element."""
        return Intrinsics(self.tensor[item])

    @property
    def device(self) -> torch.device:
        """Returns current device."""
        return self.tensor.device

    @classmethod
    def cat(
        cls,
        instances: List["Intrinsics"],
        device: Optional[torch.device] = None,
    ) -> "Intrinsics":
        """Concatenate N Intrinsics objects."""
        tensors = []
        if device is None:
            device = instances[0].tensor.device
        for inst in instances:
            tensor = inst.tensor.to(device)
            tensors.append(tensor)
        return Intrinsics(torch.cat(tensors, 0))


class Extrinsics(DataInstance):
    """Data structure for extrinsic calibration."""

    def __init__(self, tensor: torch.Tensor):
        """Init Extrinsics class.

        Args:
            tensor (torch.Tensor): (N, 4, 4) or (4, 4)
        """
        assert 2 <= len(tensor.shape) <= 3
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        self.tensor = tensor

    def to(self, device: torch.device) -> "Extrinsics":
        """Put images on device."""
        cast_tensor = self.tensor.to(device)
        return Extrinsics(cast_tensor)

    @property
    def device(self) -> torch.device:
        """Returns current device."""
        return self.tensor.device

    def __getitem__(self, item: int) -> "Extrinsics":
        """Return single element."""
        return Extrinsics(self.tensor[item])

    @classmethod
    def cat(
        cls,
        instances: List["Extrinsics"],
        device: Optional[torch.device] = None,
    ) -> "Extrinsics":
        """Concatenate N Extrinsics objects."""
        tensors = []
        if device is None:
            device = instances[0].tensor.device
        for inst in instances:
            tensor = inst.tensor.to(device)
            tensors.append(tensor)
        return Extrinsics(torch.cat(tensors, 0))


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
        """Concatenate N Images objects."""
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
        boxes2d: Optional[Sequence[Boxes2D]] = None,
        boxes3d: Optional[Sequence[Boxes3D]] = None,
        poly2d: Optional[Sequence[Poly2D]] = None,
        intrinsics: Optional[Intrinsics] = None,
        extrinsics: Optional[Extrinsics] = None,
    ) -> None:
        """Init."""
        self.metadata = metadata
        self.images = images
        assert len(metadata) == len(images)

        if boxes2d is None:
            boxes2d = [
                Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))
                for _ in range(len(images))
            ]
        self.boxes2d: Sequence[Boxes2D] = boxes2d

        if boxes3d is None:
            boxes3d = [
                Boxes3D(torch.empty(0, 8), torch.empty(0), torch.empty(0))
                for _ in range(len(images))
            ]
        self.boxes3d: Sequence[Boxes3D] = boxes3d

        if poly2d is None:
            poly2d = Poly2D()
        self.poly2d = poly2d

        if intrinsics is None:
            intrinsics = Intrinsics(
                torch.cat([torch.eye(3) for _ in range(len(images))])
            )
        self.intrinsics: Intrinsics = intrinsics

        if extrinsics is None:
            extrinsics = Extrinsics(
                torch.cat([torch.eye(4) for _ in range(len(images))])
            )
        self.extrinsics: Extrinsics = extrinsics

    def get(
        self, key: str
    ) -> Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]:
        """Get attribute by key."""
        if key in self.dict():
            value = self.dict()[key]
            return value
        raise AttributeError(f"Attribute {key} not found!")

    def dict(
        self,
    ) -> Dict[
        str, Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]
    ]:
        """Return InputSample object as dict."""
        obj_dict: Dict[
            str, Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]
        ] = {
            "metadata": self.metadata,
            "images": self.images,
            "boxes2d": self.boxes2d,
            "boxes3d": self.boxes3d,
            "poly2d": self.poly2d,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
        }
        return obj_dict

    @classmethod
    def cat(
        cls,
        instances: List["InputSample"],
        device: Optional[torch.device] = None,
    ) -> "InputSample":
        """Concatenate N InputSample objects."""
        cat_dict: Dict[
            str, Union[Sequence[Frame], DataInstance, Sequence[DataInstance]]
        ] = {}
        for k, v in instances[0].dict().items():
            if isinstance(v, list):
                cat_dict[k] = []
                for inst in instances:
                    attr = inst.get(k)
                    cat_dict[k] += [  # type: ignore
                        attr_v.to(device)
                        if isinstance(attr_v, DataInstance)
                        else attr_v
                        for attr_v in attr  # type: ignore
                    ]
            elif isinstance(v, DataInstance) and hasattr(type(v), "cat"):
                cat_dict[k] = type(v).cat(  # type: ignore
                    [inst.get(k) for inst in instances], device
                )
            else:
                raise AttributeError(
                    f"Class {type(v)} for attribute {k} must either be of type"
                    f" list or implement the cat() function (see e.g. Images)!"
                )

        return InputSample(**cat_dict)  # type: ignore

    def __getitem__(self, item: int) -> "InputSample":
        """Return single element."""
        return InputSample(
            [self.metadata[item]],
            self.images[item],
            [self.boxes2d[item]],
            [self.boxes3d[item]],
            self.intrinsics[item],
            self.extrinsics[item],
        )
