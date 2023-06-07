"""Testcases utils."""
from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import DataLoader, Dataset

from vis4d.data.loader import (
    DictDataOrList,
    VideoDataPipe,
    build_inference_dataloaders,
    default_collate,
    default_pipeline,
    multi_sensor_collate,
)
from vis4d.data.transforms import compose_batch
from vis4d.data.transforms.to_tensor import ToTensor


def get_dataloader(
    datasets: Dataset[DictDataOrList],
    batch_size: int,
    sensors: Sequence[str] | str | None = None,
) -> DataLoader[DictDataOrList]:
    """Get data loader for testing."""
    datapipe = VideoDataPipe(datasets)
    if sensors is not None:
        batchprocess_fn = compose_batch(
            [ToTensor(sensors=sensors)]  # type: ignore
        )
        collate_fn = multi_sensor_collate
    else:
        batchprocess_fn = default_pipeline  # type: ignore
        collate_fn = default_collate
    return build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=0,
        batchprocess_fn=batchprocess_fn,
        collate_fn=collate_fn,
    )[0]
