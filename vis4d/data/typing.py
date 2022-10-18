_DictStrArray = Dict[str, Tensor]
_DictStrArrayNested = Dict[str, Union[Tensor, _DictStrArray]]
DictData = Dict[str, Union[Tensor, _DictStrArrayNested]]
MultiSensorData = Dict[str, DictData]
