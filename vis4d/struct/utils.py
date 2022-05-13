"""Data structure utility functions."""


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
