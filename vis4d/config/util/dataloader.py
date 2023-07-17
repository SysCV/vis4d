"""Dataloader configuration."""
from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict, FieldReference

from vis4d.common.typing import GenericFunc
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe
from vis4d.data.loader import (
    build_inference_dataloaders,
    build_train_dataloader,
    default_collate,
    default_pipeline,
    DEFAULT_COLLATE_KEYS,
)
from vis4d.data.transforms.to_tensor import ToTensor

from .callable import get_callable_cfg


def get_train_dataloader_cfg(
    dataset_cfg: ConfigDict,
    preprocess_cfg: ConfigDict | None = None,
    data_pipe: type = DataPipe,
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    batchprocess_cfg: ConfigDict | None = None,
    collate_fn: GenericFunc = default_collate,
    collate_keys: Sequence[str] = DEFAULT_COLLATE_KEYS,
    pin_memory: bool | FieldReference = True,
    shuffle: bool | FieldReference = True,
) -> ConfigDict:
    """Creates dataloader configuration given dataset and preprocessing.

    Args:
        preprocess_cfg (ConfigDict): The configuration that contains the
            preprocessing operations.
        dataset_cfg (ConfigDict): The configuration that contains the dataset.
        samples_per_gpu (int | FieldReference, optional): How many samples each
            GPU will process. Defaults to 1.
        workers_per_gpu (int | FieldReference, optional): How many workers to
            spawn per GPU. Defaults to 1.
        data_pipe (DataPipe, optional): The data pipe class to use. Defaults to
            DataPipe.
        batchprocess_cfg (ConfigDict, optional): The configuration that
            contains the batch processing operations.
        collate_fn (GenericFunc, optional): The collate function to use.
            Defaults to default_collate.
        pin_memory (bool | FieldReference, optional): Whether to pin memory.
            Defaults to True.
        shuffle (bool | FieldReference, optional): Whether to shuffle the
            dataset. Defaults to True.

    Returns:
        ConfigDict: Configuration that can be instantiate as a dataloader.
    """
    if batchprocess_cfg is None:
        batchprocess_cfg = class_config(ToTensor)

    if preprocess_cfg is None:
        dataset = class_config(data_pipe, datasets=dataset_cfg)
    else:
        dataset = class_config(
            data_pipe, datasets=dataset_cfg, preprocess_fn=preprocess_cfg
        )

    return class_config(
        build_train_dataloader,
        dataset=dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_fn=batchprocess_cfg,
        collate_fn=get_callable_cfg(collate_fn),
        collate_keys=collate_keys,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )


def get_inference_dataloaders_cfg(
    datasets_cfg: ConfigDict | list[ConfigDict],
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    video_based_inference: bool | FieldReference = False,
    batchprocess_cfg: ConfigDict = class_config(default_pipeline),
    collate_fn: GenericFunc = default_collate,
) -> ConfigDict:
    """Creates dataloader configuration given dataset for inference.

    Args:
        datasets_cfg (ConfigDict | list[ConfigDict]): The configuration
            contains the single dataset or datasets.
        samples_per_gpu (int | FieldReference, optional): How many samples each
            GPU will process per batch. Defaults to 1.
        workers_per_gpu (int | FieldReference, optional): How many workers each
            GPU will spawn. Defaults to 1.
        video_based_inference (bool | FieldReference , optional): Whether to
            split dataset by sequences. Defaults to False.
        batchprocess_cfg (ConfigDict, optional): The config that contains the
            batch processing operations. Defaults to class_config(
                default_pipeline).
        collate_fn (GenericFunc, optional): The collate function that will be
            used to stack the batch. Defaults to default_collate.

    Returns:
        ConfigDict: The dataloader configuration.
    """
    return class_config(
        build_inference_dataloaders,
        datasets=datasets_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=video_based_inference,
        batchprocess_fn=batchprocess_cfg,
        collate_fn=get_callable_cfg(collate_fn),
    )
