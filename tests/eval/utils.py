"""Testcases utils."""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from vis4d.data.loader import (
    DictDataOrList,
    VideoDataPipe,
    build_inference_dataloaders,
)


def get_dataloader(
    datasets: Dataset[DictDataOrList], batch_size: int
) -> DataLoader[DictDataOrList]:
    """Get data loader for testing."""
    datapipe = VideoDataPipe(datasets)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=0
    )[0]
