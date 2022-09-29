"""Basic data augmentation class."""
from typing import Callable, List, Tuple

import torch

from vis4d.struct_to_revise import DictStrAny

from ..datasets.base import DataKeys, DictData


class PrettyRepMixin:
    """Creates a pretty string representation of a class with parameters."""

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        attr_str = ""
        for k, v in vars(self).items():
            if k != "type" and not k.startswith("_"):
                attr_str += f"{k}={str(v)}, "
        attr_str = attr_str.rstrip(", ")
        return f"{self.__class__.__name__}({attr_str})"


class BaseTransform(PrettyRepMixin):
    """Base transformation class."""

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (DataKeys.images,),
    ):
        """Initialize transformation."""
        super().__init__()
        self.in_keys = in_keys

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate current parameters."""
        raise NotImplementedError

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Apply transformation to given keys"""
        raise NotImplementedError


class BaseBatchTransform(PrettyRepMixin):
    """Base class for transformation applied on a batch of samples."""

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (DataKeys.images,),
    ):
        """Initialize transformation."""
        super().__init__()
        self.in_keys = in_keys

    def generate_parameters(self, data: List[DictData]) -> DictStrAny:
        """Generate current parameters."""
        raise NotImplementedError

    def __call__(
        self, data: List[DictData], parameters: DictStrAny
    ) -> List[DictData]:
        """Apply transformation to given keys"""
        raise NotImplementedError


class RandomApply(BaseTransform):
    """Apply given transform at random with given probability."""

    def __init__(
        self,
        transform: BaseTransform,
        p: float = 0.5,
        in_keys: Tuple[str, ...] = (DataKeys.images,),
    ):
        """Init."""
        super().__init__(in_keys)
        self.transform = transform
        self.p = p

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Get parameters."""
        params = self.transform.generate_parameters(data)
        params["apply"] = torch.rand(1) < self.p
        return params

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Random apply augmentation."""
        if parameters["apply"]:
            data = self.transform(data, parameters)
        return data


def transform_pipeline(
    augmentations: List[BaseTransform],
) -> Callable[[DictData], Tuple[DictData, List[DictStrAny]]]:
    """Compose transforms into single function."""

    def transform(data: DictData) -> DictData:
        params = []
        for aug in augmentations:
            param = aug.generate_parameters(data)
            params.append(param)
            data = aug(data, param)
        return data, param

    return transform


def batch_transform_pipeline(
    augmentations: List[BaseBatchTransform],
) -> Callable[[List[DictData]], List[DictData]]:
    """Compose batch transforms into single function."""

    def transform(data: List[DictData]) -> List[DictData]:
        for aug in augmentations:
            param = aug.generate_parameters(data)
            data = aug(data, param)
        return data

    return transform
