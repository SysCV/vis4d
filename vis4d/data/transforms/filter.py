"""Segmentation/Instance Mask Transform."""
from typing import List, Tuple

import torch

from vis4d.data.datasets.base import DataKeys, DictData
from vis4d.struct_to_revise import DictStrAny

from .base import Transform


def filter_by_category(
    *inputs: torch.Tensor, categories: torch.Tensor, keep: List[int]
) -> List[torch.Tensor]:
    selector = []
    for i, c in enumerate(categories):
        if c in keep:
            selector.append(i)
    outputs = []
    for x in inputs:
        outputs.append(x[selector])
    return outputs


def remap_categories(classes: torch.Tensor, mapping: List[int]):
    """Remap classes indices."""
    for i in range(len(classes)):
        classes[i] = mapping.index(classes[i])
    return classes


class FilterByCategory(Transform):
    """Filter by categories."""

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (
            DataKeys.boxes2d_classes,
            DataKeys.boxes2d,
            DataKeys.masks,
        ),
        category_key: str = DataKeys.boxes2d_classes,
        keep: List[int] = [],
    ):
        """Init."""
        super().__init__(in_keys)
        self.category_key = category_key
        self.keep = keep

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate parameters (empty)."""
        return {}

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Remap classes."""
        inputs = [data[k] for k in self.in_keys]
        outputs = filter_by_category(
            *inputs,
            categories=data[self.category_key],
            keep=self.keep,
        )
        for i, k in enumerate(self.in_keys):
            data[k] = outputs[i]
        return data


class RemapCategory(Transform):
    """Remap indices of categories."""

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (DataKeys.boxes2d_classes,),
        mapping: List[int] = [],
    ):
        """Init."""
        super().__init__(in_keys)
        self.mapping = mapping

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate parameters (empty)."""
        return {}

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Remap classes."""
        data[self.in_keys[0]] = remap_categories(
            data[self.in_keys[0]], mapping=self.mapping
        )
        return data
