"""Vis4D base module definition."""
from typing import Any, Generic, TypeVar, Union

from torch.nn import Module

from .registry import RegistryHolder

TTrainReturn = TypeVar("TTrainReturn")
TTestReturn = TypeVar("TTestReturn")

# This class will be properly typed once PEP646 is integrated to python
class Vis4DModule(Module, Generic[TTrainReturn, TTestReturn], metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Vis4D module base class."""

    def forward(self, *args: Any, **kwargs: Any) -> Union[TTrainReturn, TTestReturn]:  # type: ignore # pylint: disable=line-too-long
        """Forward."""
        raise NotImplementedError
