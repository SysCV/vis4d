"""Default dataloader configurations."""
from __future__ import annotations

from collections.abc import Callable

from ml_collections import FieldReference
from ml_collections.config_dict import ConfigDict

from vis4d.config.util import class_config
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
    default_collate,
)
from vis4d.data.transforms.base import compose_batch
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.data.typing import DictData


def get_dataloader_config(
    preprocess_cfg: ConfigDict,
    dataset_cfg: ConfigDict,
    data_pipe: type = DataPipe,
    batchprocess_cfg: ConfigDict = class_config(
        compose_batch,
        transforms=[
            class_config(PadImages),
            class_config(ToTensor),
        ],
    ),
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 4,
    train: bool = True,
    shuffle: bool | FieldReference = False,
    collate_fn: Callable[[list[DictData]], DictData] = default_collate,
) -> ConfigDict:
    """Creates dataloader configuration given dataset and preprocessing.

    Images will be padded and stacked into a batch.

    Args:
        preprocess_cfg (ConfigDict): The configuration that contains the
            preprocessing operations.
        dataset_cfg (ConfigDict): The configuration that contains the dataset.
        samples_per_gpu (int | FieldReference): How many samples each GPU will
            process. Defaults to 1.
        workers_per_gpu (int | FieldReference): How many workers to spawn per
            GPU. Defaults to 4.
        data_pipe (DataPipe): The data pipe class to use. Defaults to DataPipe.
        batchprocess_cfg (ConfigDict): The configuration that
            contains the batch processing operations.
        train (bool): Whether to create a train dataloader.
        shuffle (bool, FieldReference): Whether to shuffle the dataset.
        collate_fn (Callable): The collate function to use.

    Returns:
        ConfigDict: Configuration that can be instantiate as a dataloader.
    """
    if train:
        return class_config(
            build_train_dataloader,
            dataset=class_config(
                data_pipe,
                datasets=dataset_cfg,
                preprocess_fn=preprocess_cfg,
            ),
            batchprocess_fn=batchprocess_cfg,
            collate_fn=collate_fn,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            shuffle=shuffle,
        )

    return class_config(
        build_inference_dataloaders,
        datasets=class_config(
            data_pipe,
            datasets=dataset_cfg,
            preprocess_fn=preprocess_cfg,
        ),
        batchprocess_fn=batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        collate_fn=collate_fn,
    )
