"""Build VisT data loading pipeline."""
import itertools
import logging
from typing import List, Optional

import torch
from torch.utils import data
from scalabel.label.typing import Frame
from .samplers import TrackingInferenceSampler
from .dataset import ScalabelDataset
from .datasets import BaseDatasetConfig, build_dataset_loader
from .utils import (
    discard_labels_outside_set,
    identity_batch_collator,
    prepare_labels,
    print_class_histogram,
)


def build_train_dataset(
    train_cfg: List[BaseDatasetConfig],
) -> data.Dataset:
    """Build train dataloader with some default features."""

    # TODO save original classes
    # from scalabel.label.typing import Config as MetadataConfig
    # from scalabel.label.utils import get_leaf_categories

    dataset_loaders = [build_dataset_loader(cfg) for cfg in train_cfg]
    datasets = [ScalabelDataset(dl, True) for dl in dataset_loaders]
    train_dataset = data.ConcatDataset(datasets)
    return train_dataset


def build_test_dataset(
    dataset_cfg: BaseDatasetConfig,
) -> data.Dataset:
    """Build test dataloader with some default features."""
    dataset = ScalabelDataset(build_dataset_loader(dataset_cfg), False)
    return dataset
