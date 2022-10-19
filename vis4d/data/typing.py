"""Vis4D data types."""
from typing import Dict, Union

from torch import Tensor

_DictStrArray = Dict[str, Tensor]
_DictStrArrayNested = Dict[str, Union[Tensor, _DictStrArray]]
DictData = Dict[str, Union[Tensor, _DictStrArrayNested]]
