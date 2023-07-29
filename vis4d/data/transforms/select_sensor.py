# pylint: disable=no-member
"""Select Sensor transformation."""
from vis4d.data.typing import DictData

from .base import Transform


@Transform("data", "data")
class SelectSensor:
    """Keep data from one sensor only but keep shared data.

    Note: The input data is assumed to be in the format of DictData[DictData],
    i.e. a list of data dictionaries, each of which contains a dictionary of
    either the data from a sensor or the shared data (metadata) for all
    sensors.

    Example:
        >>> data = [
                {
                    "sensor1": {"image": 1, "label": 2},
                    "sensor2": {"image": 1, "label": 2},
                    "meta": 3},
                },
            ]
        >>> tsfm = SelectSensor(
                sensor="sensor1", sensors=["sensor1", "sensor2"]
            )
        >>> tsfm(data)
        [{"image": 1, "label": 2, "meta": 3},]
    """

    def __init__(self, selected_sensor: str) -> None:
        """Creates an instance of SelectSensor.

        Args:
            selected_sensor (str): The name of the sensor to keep.
        """
        self.selected_sensor = selected_sensor

    def __call__(self, batch: list[DictData]) -> list[DictData]:
        """Select data from one sensor only."""
        output_batch = []
        for data in batch:
            output_data = {}
            for key in data.keys():
                if key in self.sensors:  # type: ignore
                    if key == self.selected_sensor:
                        output_data.update(data[key])
                else:
                    output_data[key] = data[key]
            output_batch.append(output_data)
        return output_batch
