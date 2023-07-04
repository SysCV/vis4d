"""Test multi-sensor data connector."""

import torch

from vis4d.engine.connectors import MultiSensorDataConnector


def test_multi_sensors():
    """Test multi-sensor data connector."""
    conn = MultiSensorDataConnector(
        key_mapping={"input0": "input0"},
        sensors=["sensor0", "sensor1"],
    )
    data_dict = {
        "sensor0": {"input0": torch.rand(1, 3, 32, 32)},
        "sensor1": {"input0": torch.rand(1, 3, 32, 32)},
    }
    output = conn(data_dict)
    assert output["input0"].shape == (2, 1, 3, 32, 32)
    assert torch.allclose(output["input0"][0], data_dict["sensor0"]["input0"])


def test_single_sensor():
    """Test multi-sensor data connector for single sensor."""
    conn = MultiSensorDataConnector(
        key_mapping={"input0": "input0"},
        sensors=["sensor0"],
    )
    data_dict = {
        "sensor0": {"input0": torch.rand(1, 3, 32, 32)},
    }
    output = conn(data_dict)
    assert output["input0"].shape == (1, 3, 32, 32)
    assert torch.allclose(output["input0"][0], data_dict["sensor0"]["input0"])
