"""Vis4D data types."""
from typing import Dict, Union

from torch import Tensor

_DictStrArray = Dict[str, Tensor]
DictStrArrayNested = Dict[str, Union[Tensor, _DictStrArray]]
DictData = Dict[str, Union[Tensor, DictStrArrayNested]]
