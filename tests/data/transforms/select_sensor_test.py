#pylint: disable=unexpected-keyword-arg
"""Select Sensor transformation tests."""
from vis4d.data.transforms.select_sensor import SelectSensor


def test_select_sensor_transform() -> None:
    """Test SelectSensor transform."""
    data = [
        {
            "sensor1": {"image": 1, "label": 2},
            "sensor2": {"image": 1, "label": 2},
            "meta": 3,
        },
    ]
    tsfm = SelectSensor(
        selected_sensor="sensor1", sensors=["sensor1", "sensor2"]
    )
    assert tsfm(data) == [
        {"image": 1, "label": 2, "meta": 3},
    ]
