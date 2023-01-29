"""Default dataloader configurations."""

from __future__ import annotations

from typing import Callable

from ml_collections import FieldReference
from ml_collections.config_dict import ConfigDict

from vis4d.config.util import class_config
from vis4d.data.typing import DictData


def default_dataloader_config(
    preprocess_cfg: ConfigDict,
    dataset_cfg: ConfigDict,
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    batchprocess_fn: Callable[[list[DictData]], list[DictData]]
    | ConfigDict = lambda x: x,
    shuffle: bool | FieldReference = True,
) -> ConfigDict:
    """Creates a dataloader configuration given dataset and preprocessing.

    Args:
        preprocess_cfg (ConfigDict): The configuration that contains the
            preprocessing operations.
        dataset_cfg (ConfigDict): The configuration that contains the dataset.
        samples_per_gpu (int, optional): How many samples to load per GPU.
            Defaults to 1.
        workers_per_gpu (int, optional): How many workers to spawn per GPU.
            Defaults to 1.
        batchprocess_fn (Callable[[list[DictData]],list[DictData]] | ConfigDict
            , optional): Function to apply for each batch.
            Defaults to identity.
        shuffle (bool, optional): Whether to shuffle the dataset.

    Returns:
        ConfigDict: Configuration that can be instantiate as a dataloader.
    """
    return class_config(
        "vis4d.data.loader.build_train_dataloader",
        dataset=class_config(
            "vis4d.data.loader.DataPipe",
            datasets=dataset_cfg,
            preprocess_fn=preprocess_cfg,
        ),
        batchprocess_fn=batchprocess_fn,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        shuffle=shuffle,
    )
