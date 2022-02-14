"""Vis4D Input data structures."""

import itertools
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .structures import InputInstance


class Intrinsics(InputInstance):
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
            tensor = inst.tensor
            tensors.append(tensor)
        return Intrinsics(torch.cat(tensors, 0).to(device))

    def inverse(self) -> "Intrinsics":
        """Invert intrinsics."""
        return Intrinsics(torch.inverse(self.tensor))

    def transpose(self) -> "Intrinsics":
        """Transpose of intrinsics."""
        return Intrinsics(self.tensor.permute(0, 2, 1))

    def __len__(self) -> int:
        """Get length."""
        return len(self.tensor)


class Extrinsics(InputInstance):
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
            tensor = inst.tensor
            tensors.append(tensor)
        return Extrinsics(torch.cat(tensors, 0).to(device))

    @property
    def rotation(self) -> torch.Tensor:
        """Return (N, 3, 3) rotation matrices."""
        return self.tensor[:, :3, :3]

    @property
    def translation(self) -> torch.Tensor:
        """Return (N, 3, 1) translation vectors."""
        return self.tensor[:, :3, 3:4]

    def inverse(self) -> "Extrinsics":
        """Invert rigid transformation matrix [R^T, -R^T * t]."""
        rot = self.rotation.permute(0, 2, 1)
        t = -rot @ self.translation
        inv = torch.cat([torch.cat([rot, t], -1), self.tensor[:, 3:4]], 1)
        return Extrinsics(inv)

    def transpose(self) -> "Extrinsics":
        """Transpose of extrinsics."""
        return Extrinsics(self.tensor.permute(0, 2, 1))

    def __matmul__(
        self, other: Union["Extrinsics", torch.Tensor]
    ) -> "Extrinsics":
        """Multiply extrinsics with another extrinsics or a tensor."""
        if isinstance(other, Extrinsics):
            return Extrinsics(self.tensor @ other.tensor)
        if isinstance(other, torch.Tensor):  # pragma: no cover
            return Extrinsics(self.tensor @ other)
        raise ValueError("other must be of type Extrinsics or Tensor")

    def __len__(self) -> int:
        """Get length."""
        return len(self.tensor)


class FlipMode(Enum):
    """Enum defining the axis for horizontal / vertical flip."""

    HORIZONTAL = 3
    VERTICAL = 2


class Images(InputInstance):
    """Data structure for saving images."""

    stride: int = 32

    def __init__(
        self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]
    ):
        """Init Images class.

        Args:
            tensor (torch.Tensor): shape (N, C_1, ..., C_K, H, W) where K >= 1.
            image_sizes (list[tuple[int, int]]): Each tuple is (w, h). It can
                be smaller than (W, H) due to padding.
        """
        assert len(tensor.shape) > 3
        assert len(image_sizes) == tensor.shape[0], (
            f"Tensor shape ({tensor.shape[0]}) and image_sizes"
            f" ({len(image_sizes)}) do not match!"
        )
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

    def resize(
        self, resize_hw: Tuple[int, int], mode: str = "bilinear"
    ) -> None:
        """Resizes Images object."""
        align_corners = None if mode == "nearest" else False
        resized_ims = []
        for i in range(len(self)):
            w, h = self.image_sizes[i]
            im_t = F.interpolate(
                self.tensor[i : i + 1, ..., :h, :w],
                resize_hw,
                mode=mode,
                align_corners=align_corners,
            )
            resized_ims.append(im_t)
            self.image_sizes[i] = (im_t.shape[3], im_t.shape[2])
        self.tensor = torch.cat(resized_ims)

    def flip(self, mode: FlipMode = FlipMode.HORIZONTAL) -> None:
        """Flips Images object."""
        for i in range(len(self)):
            w, h = self.image_sizes[i]
            self.tensor[i : i + 1, ..., :h, :w] = self.tensor[
                i : i + 1, ..., :h, :w
            ].flip(mode.value)


class PointCloud(InputInstance):
    """Data structure for loading point cloud."""

    num_point_feature: int = 4

    def __init__(self, tensor: torch.tensor):
        """Init PointCloud class.

        Args:
            tensor (torch.Tensor): (N, C) or (B, N, C)
        """
        assert 2 <= len(tensor.shape) <= 3
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        self.tensor = tensor

    def __len__(self) -> int:
        """Return number of PoinCloud."""
        return int(self.tensor.shape[0])

    def __getitem__(self, idx: int) -> "PointCloud":
        """Access single points."""
        return PointCloud(self.tensor[idx])

    def to(self, device: torch.device) -> "PointCloud":
        """Put PoinCloud on device."""
        cast_tensor = self.tensor.to(device)
        return PointCloud(cast_tensor)

    @classmethod
    def cat(
        cls,
        instances: List["PointCloud"],
        device: Optional[torch.device] = None,
    ) -> "PointCloud":
        """Concatenate N PointCloud objects into Padded foramt.

        Returns:
            Tensor: [Batch, N_max, num_point_feature].
        """
        assert isinstance(instances, (list, tuple))
        assert len(instances) > 0

        if device is None:
            device = instances[0].tensor.device

        max_points = max([p.tensor.shape[1] for p in instances])
        max_feature = max([p.tensor.shape[2] for p in instances])

        tot_batch = sum([len(inst) for inst in instances])

        pad_points = torch.zeros(
            (tot_batch, max_points, max_feature), device=device
        )

        cum_len = 0
        for inst in instances:
            cur_len = inst.tensor.shape[0]
            pad_points[
                cum_len : cum_len + cur_len, : inst.tensor.shape[1], :
            ] = inst.tensor
            cum_len += cur_len
        return PointCloud(pad_points)

    @property
    def device(self) -> torch.device:
        """Returns current device."""
        return self.tensor.device
