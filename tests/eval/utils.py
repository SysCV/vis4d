"""Testcases utils."""
from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import DataLoader, Dataset

from vis4d.data.data_pipe import DataPipe
from vis4d.data.loader import build_inference_dataloaders, multi_sensor_collate
from vis4d.data.transforms import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.data.typing import DictDataOrList


def get_dataloader(
    datasets: Dataset[DictDataOrList],
    batch_size: int,
    sensors: Sequence[str] | str | None = None,
) -> DataLoader[DictDataOrList]:
    """Get data loader for testing."""
    datapipe = DataPipe(datasets)

    if sensors is not None:
        return build_inference_dataloaders(
            datapipe,
            samples_per_gpu=batch_size,
            workers_per_gpu=0,
            batchprocess_fn=compose([ToTensor(sensors=sensors)]),  # type: ignore # pylint: disable=line-too-long
            collate_fn=multi_sensor_collate,
        )[0]

    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=0
    )[0]
