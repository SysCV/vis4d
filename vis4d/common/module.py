"""Vis4D base module definition."""
from __future__ import annotations

import abc
import copy
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

from torch.nn import Module

from .registry import RegistryHolder

if TYPE_CHECKING:  # pragma: no cover
    from vis4d.struct import ModuleCfg

TTrainReturn = TypeVar("TTrainReturn")
TTestReturn = TypeVar("TTestReturn")

# This class will be properly typed once PEP646 is integrated to python
class Vis4DModule(
    Module, Generic[TTrainReturn, TTestReturn], metaclass=RegistryHolder  # type: ignore  # pylint: disable=line-too-long
):
    """Vis4D module base class."""

    def forward(  # type: ignore
        self, *args: Any, **kwargs: Any
    ) -> Union[TTrainReturn, TTestReturn]:
        """Call __call__ in forward to enable mypy compatibility."""
        return self.__call__(*args, **kwargs)  # pragma: no cover

    @abc.abstractmethod
    def __call__(  # type: ignore
        self, *args: Any, **kwargs: Any
    ) -> Union[TTrainReturn, TTestReturn]:
        """Forward pass implementation."""
        raise NotImplementedError


def build_module(  # type: ignore
    cfg: ModuleCfg, bound: Any
) -> Vis4DModule[TTrainReturn, TTestReturn]:
    """Build a module from config."""
    registry = RegistryHolder.get_registry(bound)
    cfg = copy.deepcopy(cfg)
    module_type = cfg.pop("type", None)
    if module_type is None:
        raise ValueError(f"Need type argument in module config: {cfg}")
    if module_type in registry:
        module = registry[module_type](**cfg)
        assert isinstance(module, Vis4DModule)
        return module
    raise NotImplementedError(f"Module {module_type} not found.")
