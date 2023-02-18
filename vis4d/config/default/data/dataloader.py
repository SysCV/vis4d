"""Default dataloader configurations."""

from __future__ import annotations

from typing import Callable

from ml_collections import FieldReference
from ml_collections.config_dict import ConfigDict

from vis4d.common.distributed import get_world_size
from vis4d.config.util import class_config
from vis4d.data.loader import DataPipe, build_train_dataloader
from vis4d.data.typing import DictData


def default_image_dl(
    preprocess_cfg: ConfigDict,
    dataset_cfg: ConfigDict,
    batch_size: int | FieldReference = 1,
    num_workers_per_gpu: int | FieldReference = 4,
    shuffle: bool | FieldReference = True,
) -> ConfigDict:
    """Creates a dataloader configuration given dataset and preprocessing.

    Images will be padded and stacked into a batch.

    Args:
        preprocess_cfg (ConfigDict): The configuration that contains the
            preprocessing operations.
        dataset_cfg (ConfigDict): The configuration that contains the dataset.
        batch_size (int | FieldReference,  optional): Batch size.
            Each GPU will process batch_size / n_gpu samples. Defaults to 1.
        num_workers_per_gpu (int | FieldReference, optional): How many workers
            to spawn per GPU. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the dataset.

    Returns:
        ConfigDict: Configuration that can be instantiate as a dataloader.
    """
    n_gpus = get_world_size()

    return default_dataloader_config(
        preprocess_cfg,
        dataset_cfg,
        batch_size // n_gpus,
        num_workers_per_gpu,
        class_config("vis4d.data.transforms.pad.pad_image"),
        # FIXME: Currently, resolving transforms is broken if we directly pass
        # the function instead of the name to resolve, since resolving
        # the function path with the decorator converts e.g. 'pad_image' which
        # is 'BatchTransform.__call__.<locals>.get_transform_fn'
        # to  vis4d.data.transforms.base.get_transform_fn.
        # We need to use to full config path for now. Should probably be fixed
        # with the transform update
        shuffle,
    )


def default_dataloader_config(
    preprocess_cfg: ConfigDict,
    dataset_cfg: ConfigDict,
    samples_per_gpu: int | FieldReference = 0,
    workers_per_gpu: int | FieldReference = 0,
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
        build_train_dataloader,
        dataset=class_config(
            DataPipe,
            datasets=dataset_cfg,
            preprocess_fn=preprocess_cfg,
        ),
        batchprocess_fn=batchprocess_fn,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        shuffle=shuffle,
    )
