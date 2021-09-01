"""Build VisT data loading pipeline."""
import os
import random
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils import data

from ..common.utils import get_rank
from .dataset import ScalabelDataset
from .datasets import (
    BaseDatasetConfig,
    BaseDatasetLoader,
    build_dataset_loader,
)
from .samplers import TrackingInferenceSampler
from .utils import identity_batch_collator


def build_dataset_loaders(
    train_cfg: List[BaseDatasetConfig],
    test_cfg: List[BaseDatasetConfig],
    input_dir: Optional[str] = None,
) -> Tuple[
    List[BaseDatasetLoader], List[BaseDatasetLoader], List[BaseDatasetLoader]
]:
    """Build dataset loaders."""
    train_loaders = [build_dataset_loader(cfg) for cfg in train_cfg]
    test_loaders = [build_dataset_loader(cfg) for cfg in test_cfg]
    predict_loaders = []
    if input_dir is not None:
        if input_dir[-1] == "/":
            input_dir = input_dir[:-1]
        dataset_name = os.path.basename(input_dir)
        predict_loaders += [
            build_dataset_loader(
                BaseDatasetConfig(
                    type="Custom",
                    name=dataset_name,
                    data_root=input_dir,
                )
            )
        ]
    return train_loaders, test_loaders, predict_loaders


class VisTDataModule(pl.LightningDataModule):
    """Data module for VisT."""

    def __init__(
        self,
        samples_per_gpu: int,
        workers_per_gpu: int,
        train_loaders: List[BaseDatasetLoader],
        test_loaders: List[BaseDatasetLoader],
        predict_loaders: List[BaseDatasetLoader],
        category_mapping: Optional[Dict[str, int]] = None,
        image_channel_mode: str = "RGB",
        seed: Optional[int] = None,
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        assert not (
            len(train_loaders)
            == len(test_loaders)
            == len(predict_loaders)
            == 0
        ), "Please specify either train, test or predict datasets."
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.predict_loaders = predict_loaders
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.category_mapping = category_mapping
        self.image_channel_mode = image_channel_mode
        self.seed = seed
        self.train_datasets = None  # type: Optional[List[ScalabelDataset]]
        self.test_datasets = None  # type: Optional[List[ScalabelDataset]]
        self.predict_datasets = None  # type: Optional[List[ScalabelDataset]]

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize dataset classes."""
        if len(self.train_loaders) > 0:
            self.train_datasets = [
                ScalabelDataset(
                    dl, True, self.category_mapping, self.image_channel_mode
                )
                for dl in self.train_loaders
            ]

        if len(self.test_loaders) > 0:
            self.test_datasets = [
                ScalabelDataset(
                    dl, False, self.category_mapping, self.image_channel_mode
                )
                for dl in self.test_loaders
            ]

        if len(self.predict_loaders) > 0:
            self.predict_datasets = [
                ScalabelDataset(
                    dl, False, self.category_mapping, self.image_channel_mode
                )
                for dl in self.predict_loaders
            ]

    def train_dataloader(self) -> data.DataLoader:
        """Return dataloader for training."""
        train_dataset = data.ConcatDataset(self.train_datasets)

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.workers_per_gpu,
                rank=get_rank(),
                seed=self.seed,
            )
            if self.seed is not None
            else None
        )

        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=self.samples_per_gpu,
            num_workers=self.workers_per_gpu,
            collate_fn=identity_batch_collator,
            worker_init_fn=init_fn,
            persistent_workers=self.workers_per_gpu > 0,
        )
        return train_dataloader

    def predict_dataloader(
        self,
    ) -> Union[data.DataLoader, List[data.DataLoader]]:
        """Return dataloader(s) for prediction."""
        if self.predict_datasets is not None:
            return self._build_inference_dataloaders(self.predict_datasets)
        return self.test_dataloader()

    def val_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for validation."""
        return self.test_dataloader()

    def test_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for testing."""
        assert self.test_datasets is not None
        return self._build_inference_dataloaders(self.test_datasets)

    def _build_inference_dataloaders(
        self, datasets: List[ScalabelDataset]
    ) -> List[data.DataLoader]:
        """Build dataloaders for test / predict."""
        dataloaders = []
        for dataset in datasets:
            sampler = None  # type: Optional[data.Sampler]
            if dataset.cfg.inference_sampling == "sequence_based":
                sampler = TrackingInferenceSampler(dataset)

            test_dataloader = data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.workers_per_gpu,
                sampler=sampler,
                collate_fn=identity_batch_collator,
                persistent_workers=self.workers_per_gpu > 0,
            )
            dataloaders.append(test_dataloader)
        return dataloaders


def worker_init_fn(
    worker_id: int, num_workers: int, rank: int, seed: int
) -> None:  # pragma: no cover
    """Init worker with unique seed.

    The seed of each worker equals to:
    num_worker * rank + worker_id + user_seed
    """
    worker_seed = num_workers * rank + worker_id + seed
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
