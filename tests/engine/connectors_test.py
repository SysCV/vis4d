"""Test data connector."""

import torch

from vis4d.engine.connectors import MultiSensorDataConnector, data_key


def test_multi_sensors():
    """Test multi-sensor data connector."""
    conn = MultiSensorDataConnector(
        key_mapping={
            "metadata": "metadata",
            "input0": data_key("input0", sensors=["sensor0", "sensor1"]),
        }
    )
    data_dict = {
        "metadata": torch.rand(1),
        "sensor0": {"input0": torch.rand(1, 3, 32, 32)},
        "sensor1": {"input0": torch.rand(1, 3, 32, 32)},
    }
    output = conn(data_dict)

    assert len(output["input0"]) == 2
    assert torch.allclose(output["metadata"], data_dict["metadata"])
    assert torch.allclose(output["input0"][0], data_dict["sensor0"]["input0"])
    assert torch.allclose(output["input0"][1], data_dict["sensor1"]["input0"])


def test_single_sensor():
    """Test multi-sensor data connector for single sensor."""
    conn = MultiSensorDataConnector(
        key_mapping={
            "metadata": "metadata",
            "input0": data_key("input0", sensors=["sensor0"]),
        }
    )
    data_dict = {
        "metadata": torch.rand(1),
        "sensor0": {"input0": torch.rand(1, 3, 32, 32)},
    }
    output = conn(data_dict)

    assert len(output["input0"]) == 1
    assert torch.allclose(output["metadata"], data_dict["metadata"])
    assert torch.allclose(output["input0"][0], data_dict["sensor0"]["input0"])
