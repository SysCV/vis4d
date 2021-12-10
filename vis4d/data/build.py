"""Build Vis4D data loading pipeline."""
import os
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pydantic import BaseModel
from torch.utils import data

from ..common.registry import RegistryHolder
from ..common.utils import get_world_size
from ..struct import InputSample
from .dataset import ScalabelDataset
from .datasets import (
    BaseDatasetConfig,
    BaseDatasetLoader,
    build_dataset_loader,
)
from .samplers import (
    BaseSamplerConfig,
    TrackingInferenceSampler,
    build_data_sampler,
)
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


class DataModuleConfig(BaseModel):
    """Config for Default data module in Vis4D."""

    type: str = "Vis4DDataModule"
    pin_memory: bool = False
    train_sampler: Optional[BaseSamplerConfig]
    category_mapping: Optional[Dict[str, Dict[str, int]]]


def build_category_mappings(
    cfg: DataModuleConfig, model_category_mapping: Optional[Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    """Build category mappings."""
    if cfg.category_mapping is not None:
        pass
    assert model_category_mapping is not None
    return {"all": model_category_mapping}


class Vis4DDataModule(pl.LightningDataModule, metaclass=RegistryHolder):
    """Default Data module for Vis4D."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        samples_per_gpu: int,
        workers_per_gpu: int,
        train_loaders: List[BaseDatasetLoader],
        test_loaders: List[BaseDatasetLoader],
        predict_loaders: List[BaseDatasetLoader],
        category_mapping: Optional[Dict[str, int]] = None,
        image_channel_mode: str = "RGB",
        seed: Optional[int] = None,
        cfg: DataModuleConfig = DataModuleConfig(),
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        assert not (
            len(train_loaders)
            == len(test_loaders)
            == len(predict_loaders)
            == 0
        ), "Please specify either train, test or predict datasets."
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.category_mapping = category_mapping
        self.image_channel_mode = image_channel_mode
        self.seed = seed
        self.pin_memory = cfg.pin_memory
        self.train_datasets: Optional[List[ScalabelDataset]] = None
        self.test_datasets: Optional[List[ScalabelDataset]] = None
        self.predict_datasets: Optional[List[ScalabelDataset]] = None
        self.train_sampler = cfg.train_sampler
        if len(train_loaders) > 0:
            self.train_datasets = [
                ScalabelDataset(
                    dl, True, self.category_mapping, self.image_channel_mode
                )
                for dl in train_loaders
            ]

        if len(test_loaders) > 0:
            self.test_datasets = [
                ScalabelDataset(
                    dl, False, self.category_mapping, self.image_channel_mode
                )
                for dl in test_loaders
            ]

        if len(predict_loaders) > 0:
            self.predict_datasets = [
                ScalabelDataset(
                    dl, False, self.category_mapping, self.image_channel_mode
                )
                for dl in predict_loaders
            ]

    def train_dataloader(self) -> data.DataLoader:
        """Return dataloader for training."""
        assert self.train_datasets is not None
        if self.train_sampler is not None:
            train_sampler: Optional[
                data.Sampler[List[int]]
            ] = build_data_sampler(
                self.train_sampler, self.train_datasets, self.samples_per_gpu
            )
            batch_size, shuffle = 1, False
        else:
            train_sampler = None
            batch_size, shuffle = self.samples_per_gpu, True
        train_dataset = data.ConcatDataset(self.train_datasets)
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            batch_size=batch_size,
            num_workers=self.workers_per_gpu,
            collate_fn=identity_batch_collator,
            persistent_workers=self.workers_per_gpu > 0,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
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

    def transfer_batch_to_device(
        self,
        batch: List[List[InputSample]],
        device: torch.device,
        dataloader_idx: int,
    ) -> List[InputSample]:
        """Put input in correct format for model, move to device."""
        # group by ref views by sequence: NxM --> MxN, where M=num_refs, N=BS
        batch = [
            [batch[j][i] for j in range(len(batch))]
            for i in range(len(batch[0]))
        ]
        return [InputSample.cat(elem, device) for elem in batch]

    def _build_inference_dataloaders(
        self, datasets: List[ScalabelDataset]
    ) -> List[data.DataLoader]:
        """Build dataloaders for test / predict."""
        dataloaders = []
        for dataset in datasets:
            sampler: Optional[data.Sampler] = None
            if get_world_size() > 1 and dataset.has_sequences:
                sampler = TrackingInferenceSampler(dataset)  # pragma: no cover

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


def build_data_module(
    samples_per_gpu: int,
    workers_per_gpu: int,
    train_loaders: List[BaseDatasetLoader],
    test_loaders: List[BaseDatasetLoader],
    predict_loaders: List[BaseDatasetLoader],
    category_mapping: Optional[Dict[str, int]] = None,
    image_channel_mode: str = "RGB",
    seed: Optional[int] = None,
    cfg: DataModuleConfig = DataModuleConfig(),
) -> Vis4DDataModule:
    """Build a sampler."""
    registry = RegistryHolder.get_registry(Vis4DDataModule)
    registry["Vis4DDataModule"] = Vis4DDataModule
    if cfg.type in registry:
        module = registry[cfg.type](
            samples_per_gpu,
            workers_per_gpu,
            train_loaders,
            test_loaders,
            predict_loaders,
            category_mapping,
            image_channel_mode,
            seed,
            cfg,
        )
        assert isinstance(module, Vis4DDataModule)
        return module
    raise NotImplementedError(f"Sampler {cfg.type} not known!")
