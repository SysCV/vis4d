"""Default dataloader configurations."""
from __future__ import annotations

from ml_collections import FieldReference

from vis4d.config import ConfigDict, class_config, delay_instantiation
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
    default_collate,
    default_pipeline,
)


def get_train_dataloader_cfg(
    preprocess_cfg: ConfigDict,
    dataset_cfg: ConfigDict,
    data_pipe: type = DataPipe,
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    batchprocess_cfg: ConfigDict = class_config(default_pipeline),
    collate_fn: ConfigDict = delay_instantiation(
        class_config(default_collate)
    ),
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
        collate_fn (ConfigDict, optional): The collate function to use.
            Defaults to delay_instantiation(class_config(default_collate)).
        pin_memory (bool | FieldReference, optional): Whether to pin memory.
            Defaults to True.
        shuffle (bool | FieldReference, optional): Whether to shuffle the
            dataset. Defaults to True.

    Returns:
        ConfigDict: Configuration that can be instantiate as a dataloader.
    """
    return class_config(
        build_train_dataloader,
        dataset=class_config(
            data_pipe,
            datasets=dataset_cfg,
            preprocess_fn=preprocess_cfg,
        ),
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_fn=batchprocess_cfg,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )


def get_inference_dataloaders_cfg(
    datasets_cfg: ConfigDict | list[ConfigDict],
    samples_per_gpu: int | FieldReference = 1,
    workers_per_gpu: int | FieldReference = 1,
    video_based_inference: bool | FieldReference = False,
    batchprocess_cfg: ConfigDict = class_config(default_pipeline),
    collate_fn: ConfigDict = delay_instantiation(
        class_config(default_collate)
    ),
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
        collate_fn (ConfigDict, optional): The collate function that will be
            used to stack the batch. Defaults to delay_instantiation(
                class_config(default_collate)).

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
        collate_fn=collate_fn,
    )
