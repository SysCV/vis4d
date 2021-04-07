"""Base class for meta architectures."""

import abc
from typing import Dict, List, Tuple, Union

import torch

from openmt.config import Config
from openmt.core.registry import RegistryHolder
from openmt.struct import Boxes2D


class BaseMetaArch(torch.nn.Module, metaclass=RegistryHolder):
    """Base model class."""

    def forward(
        self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Union[Dict[str, torch.Tensor], List[Boxes2D]]:
        """Model forward function."""
        if self.training:
            return self.forward_train(batch_inputs)
        return self.forward_test(batch_inputs)

    @abc.abstractmethod
    def forward_train(
        self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> List[Boxes2D]:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        raise NotImplementedError


def build_model(cfg: Config) -> BaseMetaArch:
    """Build the whole model architecture using meta_arch templates.

    Note that it does not load any weights from ``cfg``.
    """
    assert cfg.tracking is not None
    registry = RegistryHolder.get_registry(__package__)
    if cfg.tracking.type in registry:
        module = registry[cfg.tracking.type](cfg)
        assert isinstance(module, BaseMetaArch)
        return module
    raise NotImplementedError(
        f"Meta architecture {cfg.tracking.type} not found."
    )
