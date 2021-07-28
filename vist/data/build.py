"""Build VisT data loading pipeline."""
from typing import List, Dict, Optional

from torch.utils import data
from .dataset import ScalabelDataset
from .datasets import BaseDatasetConfig, build_dataset_loader


def build_train_dataset(
    train_cfg: List[BaseDatasetConfig],
    cats_name2id: Optional[Dict[str, int]] = None,
) -> data.Dataset:
    """Build train dataloader with some default features."""
    dataset_loaders = [build_dataset_loader(cfg) for cfg in train_cfg]
    datasets = [ScalabelDataset(dl, True, cats_name2id) for dl in dataset_loaders]
    train_dataset = data.ConcatDataset(datasets)
    return train_dataset


def build_test_dataset(
    dataset_cfg: BaseDatasetConfig,
    cats_name2id: Optional[Dict[str, int]] = None,
) -> ScalabelDataset:
    """Build test dataloader with some default features."""
    dataset = ScalabelDataset(build_dataset_loader(dataset_cfg), False, cats_name2id)
    return dataset
