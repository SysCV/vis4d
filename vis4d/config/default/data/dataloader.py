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
from vis4d.data.typing import DictData


def default_image_dataloader(
    preprocess_cfg: ConfigDict,
    dataset_cfg: ConfigDict,
    num_samples_per_gpu: int | FieldReference = 1,
    num_workers_per_gpu: int | FieldReference = 4,
    shuffle: bool | FieldReference = False,
    # TODO: Currently, resolving transforms is broken if we directly pass
    # the function instead of the name to resolve, since resolving
    # the function path with the decorator converts e.g. 'pad_image' which
    # is 'BatchTransform.__call__.<locals>.get_transform_fn'
    # to  vis4d.data.transforms.base.get_transform_fn.
    # We need to use to full config path for now. Should probably be fixed
    # with the transform update
    batchprocess_cfg: ConfigDict = class_config(
        "vis4d.data.transforms.pad.pad_image"
    ),
    data_pipe: type = DataPipe,
    train: bool = True,
    collate_fn: Callable[[list[DictData]], DictData] = default_collate,
) -> ConfigDict:
    """Creates a dataloader configuration given dataset and preprocessing.

    Images will be padded and stacked into a batch.

    Args:
        preprocess_cfg (ConfigDict): The configuration that contains the
            preprocessing operations.
        dataset_cfg (ConfigDict): The configuration that contains the dataset.
        num_samples_per_gpu (int | FieldReference): How many samples
            each GPU will process. Defaults to 1.
        num_workers_per_gpu (int | FieldReference): How many workers
            to spawn per GPU. Defaults to 4.
        shuffle (bool, FieldReference): Whether to shuffle the dataset.
        batchprocess_cfg (ConfigDict): The configuration that
            contains the batch processing operations.
        data_pipe (Callable): The data pipe class to use.
            Defaults to DataPipe.
        train (bool): Whether to create a train dataloader.
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
            samples_per_gpu=num_samples_per_gpu,
            workers_per_gpu=num_workers_per_gpu,
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
        samples_per_gpu=num_samples_per_gpu,
        workers_per_gpu=num_workers_per_gpu,
        collate_fn=collate_fn,
    )
