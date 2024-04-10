"""Common functionality for seg tests."""

from torch.utils.data import DataLoader, Dataset

from vis4d.data.data_pipe import DataPipe
from vis4d.data.loader import (
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms import mask, normalize, resize
from vis4d.data.transforms.base import compose
from vis4d.data.typing import DictDataOrList


def get_train_dataloader(
    datasets: Dataset[DictDataOrList], batch_size: int
) -> DataLoader[DictDataOrList]:
    """Get data loader for training."""
    preprocess_fn = compose(
        [
            resize.GenResizeParameters((64, 64)),
            resize.ResizeImages(),
            resize.ResizeInstanceMasks(),
            normalize.NormalizeImages(),
            mask.ConvertInstanceMaskToSegMask(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )


def get_test_dataloader(
    datasets: Dataset[DictDataOrList], batch_size: int
) -> DataLoader[DictDataOrList]:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            resize.GenResizeParameters((64, 64)),
            resize.ResizeImages(),
            normalize.NormalizeImages(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )[0]
