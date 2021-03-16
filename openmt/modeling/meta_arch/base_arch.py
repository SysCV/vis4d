"""Base class for meta architectures."""

import abc
from typing import Dict, Tuple

import torch

from openmt.core.registry import RegistryHolder


class BaseMetaArch(metaclass=RegistryHolder):
    @abc.abstractmethod
    def forward(self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]):
        """Process proposals and output predictions and possibly target
        assignments."""
        raise NotImplementedError
