"""Common functionality for segment tests."""
from vis4d.data.loader import (
    _DATALOADER,
    _DATASET,
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms import mask, normalize, resize
from vis4d.data.transforms.base import compose


def get_train_dataloader(datasets: _DATASET, batch_size: int) -> _DATALOADER:
    """Get data loader for training."""
    preprocess_fn = compose(
        [
            resize.GenerateResizeParameters((64, 64)),
            resize.ResizeImage(),
            resize.ResizeInstanceMasks(),
            normalize.NormalizeImage(),
            mask.ConvertInstanceMaskToSegmentationMask(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )


def get_test_dataloader(datasets: _DATASET, batch_size: int) -> _DATALOADER:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            resize.GenerateResizeParameters((64, 64)),
            resize.ResizeImage(),
            normalize.NormalizeImage(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )[0]
