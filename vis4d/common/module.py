"""Vis4D base module definition."""
import abc
from typing import Any, Generic, TypeVar, Union

from torch.nn import Module

from .registry import RegistryHolder

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
