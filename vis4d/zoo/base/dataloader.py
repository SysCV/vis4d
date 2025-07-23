"""Dataloader configuration."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict, FieldReference

from vis4d.common.typing import GenericFunc
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe
from vis4d.data.loader import (
    DEFAULT_COLLATE_KEYS,
    build_inference_dataloaders,
    build_train_dataloader,
    default_collate,
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
    sensors: Sequence[str] | None = None,
    pin_memory: bool | FieldReference = True,
    shuffle: bool | FieldReference = True,
    aspect_ratio_grouping: bool | FieldReference = False,
) -> ConfigDict:
    """Creates dataloader configuration given dataset and preprocessing.

    Args:
        dataset_cfg (ConfigDict): The configuration that contains the dataset.
        preprocess_cfg (ConfigDict): The configuration that contains the
            preprocessing operations. Defaults to None. If None, no
            preprocessing will be applied.
        samples_per_gpu (int | FieldReference, optional): How many samples each
            GPU will process. Defaults to 1.
        workers_per_gpu (int | FieldReference, optional): How many workers to
            spawn per GPU. Defaults to 1.
        data_pipe (DataPipe, optional): The data pipe class to use. Defaults to
            DataPipe.
        batchprocess_cfg (ConfigDict, optional): The config that contains the
            batch processing operations. Defaults to None. If None, ToTensor
            will be used.
        collate_fn (GenericFunc, optional): The collate function to use.
            Defaults to default_collate.
        collate_keys (Sequence[str], optional): The keys to collate. Defaults
            to DEFAULT_COLLATE_KEYS.
        sensors (Sequence[str], optional): The sensors to collate. Defaults to
            None.
        pin_memory (bool | FieldReference, optional): Whether to pin memory.
            Defaults to True.
        shuffle (bool | FieldReference, optional): Whether to shuffle the
            dataset. Defaults to True.
        aspect_ratio_grouping (bool | FieldReference, optional): Whether to
            group the samples by aspect ratio. Defaults to False.

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
        sensors=sensors,
        pin_memory=pin_memory,
        shuffle=shuffle,
        aspect_ratio_grouping=aspect_ratio_grouping,
    )


def get_inference_dataloaders_cfg(
    datasets_cfg: ConfigDict | list[ConfigDict],
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    video_based_inference: bool | FieldReference = False,
    batchprocess_cfg: ConfigDict | None = None,
    collate_fn: GenericFunc = default_collate,
    collate_keys: Sequence[str] = DEFAULT_COLLATE_KEYS,
    sensors: Sequence[str] | None = None,
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
            batch processing operations. Defaults to None. If None, ToTensor
            will be used.
        collate_fn (GenericFunc, optional): The collate function that will be
            used to stack the batch. Defaults to default_collate.
        collate_keys (Sequence[str], optional): The keys to collate. Defaults
            to DEFAULT_COLLATE_KEYS.
        sensors (Sequence[str], optional): The sensors to collate. Defaults to
            None.

    Returns:
        ConfigDict: The dataloader configuration.
    """
    if batchprocess_cfg is None:
        batchprocess_cfg = class_config(ToTensor)

    return class_config(
        build_inference_dataloaders,
        datasets=datasets_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=video_based_inference,
        batchprocess_fn=batchprocess_cfg,
        collate_fn=get_callable_cfg(collate_fn),
        collate_keys=collate_keys,
        sensors=sensors,
    )
