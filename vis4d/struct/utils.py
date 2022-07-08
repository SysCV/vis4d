"""Data structure utility functions."""
from typing import Optional

import torch

from .sample import InputData


def clone(data) -> InputData:
    """Create a copy of the object."""
    class_ids = self.class_ids.clone() if self.class_ids is not None else None
    track_ids = self.track_ids.clone() if self.track_ids is not None else None
    return type(self)(self.boxes.clone(), class_ids, track_ids)


def to_device(data, device: torch.device):
    """Move data to given device."""

    def value_to_device(value):
        if isinstance(value, dict):
            new_value = to_device(value, device)
        elif isinstance(value, (list, tuple)):
            new_value = []
            for v in value:
                new_value += [to_device(v, device)]
            if isinstance(value, tuple):
                new_value = tuple(new_value)
        elif isinstance(torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = value
        return new_value

    new_dict = {}
    for key, val in data.items():
        new_dict[k] = value_to_device(val)
    return new_dict


def batch_input(
    instances: List[InputData],
    device: Optional[torch.device] = None,
) -> InputData:
    """Concatenate N InputData dictionaries."""

    cat_dict: Dict[str, InputSampleData] = {}
    for k, v in instances[0].dict().items():
        if isinstance(v, list):
            assert len(v) > 0, "Do not input empty inputSamples to .cat!"
            attr_list = []
            if isinstance(v[0], dict):
                for inst in instances:
                    attr_v = inst.get(k)
                    for item in attr_v:
                        assert isinstance(item, dict)
                        attr_list += [
                            {k: v.to(device) for k, v in item.items()}
                        ]
            else:
                for inst in instances:
                    attr_v = inst.get(k)
                    assert isinstance(attr_v, list)
                    attr_list += attr_v  # type: ignore
            cat_dict[k] = attr_list
        elif isinstance(v, InputInstance):
            cat_dict[k] = type(v).cat(
                [inst.get(k) for inst in instances], device  # type: ignore
            )
        else:
            raise AttributeError(
                f"Class {type(v)} for attribute {k} must be of type list "
                "or InputInstance!"
            )

    return InputSample(**cat_dict)  # type: ignore
