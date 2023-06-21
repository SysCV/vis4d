"""ToTensor transformation."""
import numpy as np
import torch

from vis4d.data.const import CommonKeys as K
from vis4d.data.typing import DictData

from .base import Transform


def _replace_arrays(data: DictData) -> None:
    """Replace numpy arrays with tensors."""
    for key in data.keys():
        if key in [K.images, K.original_images]:
            data[key] = torch.from_numpy(data[key]).permute(0, 3, 1, 2)
        elif isinstance(data[key], np.ndarray):
            data[key] = torch.from_numpy(data[key])
        elif isinstance(data[key], dict):
            _replace_arrays(data[key])
        elif isinstance(data[key], list):
            for i, entry in enumerate(data[key]):
                if isinstance(entry, np.ndarray):
                    data[key][i] = torch.from_numpy(entry)


@Transform("data", "data")
class ToTensor:
    """Transform all entries in a list of DataDict from numpy to torch.

    Note that we reshape K.images from NHWC to NCHW.
    """

    def __call__(self, batch: list[DictData]) -> list[DictData]:
        """Transform all entries to tensor."""
        for data in batch:
            _replace_arrays(data)
        return batch


@Transform("data", "data")
class SelectSensor:
    """Keep data from one sensor only."""

    def __init__(self, sensor: str):
        """Creates an instance of SelectSensor.

        Args:
            sensor (str): Sensor name.
        """
        self.sensor = sensor

    def __call__(self, batch: list[DictData]) -> list[DictData]:
        """Select data from one sensor only."""
        output_batch = []
        for data in batch:
            output_data = {}
            for key in data.keys():
                if isinstance(data[key], dict):
                    if key == self.sensor:
                        output_data.update(data[key])
                else:
                    output_data[key] = data[key]
            output_batch.append(output_data)
        return output_batch
