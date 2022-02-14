"""Data module composing the data loading pipeline."""
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from ..common.registry import RegistryHolder
from ..common.utils import get_world_size
from ..struct import InputSample
from .dataset import ScalabelDataset
from .handler import Vis4DDatasetHandler
from .samplers import BaseSampler, TrackingInferenceSampler
from .utils import identity_batch_collator


class Vis4DDataModule(pl.LightningDataModule, metaclass=RegistryHolder):
    """Default data module for Vis4D."""

    def __init__(
        self,
        samples_per_gpu: int,
        workers_per_gpu: int,
        train_datasets: Optional[Vis4DDatasetHandler] = None,
        test_datasets: Optional[List[Vis4DDatasetHandler]] = None,
        predict_datasets: Optional[List[Vis4DDatasetHandler]] = None,
        seed: Optional[int] = None,
        pin_memory: bool = False,
        train_sampler: Optional[BaseSampler] = None,
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.seed = seed
        self.pin_memory = pin_memory
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.predict_datasets = predict_datasets
        self.train_sampler = train_sampler

    def train_dataloader(self) -> data.DataLoader:
        """Return dataloader for training."""
        assert self.train_datasets is not None, "No train datasets specified!"
        if self.train_sampler is not None:
            batch_size, shuffle = 1, False
        else:
            batch_size, shuffle = self.samples_per_gpu, True
        train_dataloader = data.DataLoader(
            self.train_datasets,
            batch_sampler=self.train_sampler,
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
        return self.test_dataloader()  # pragma: no cover

    def val_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for validation."""
        return self.test_dataloader()

    def test_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for testing."""
        assert self.test_datasets is not None, "No test datasets specified!"
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
            assert (
                len(dataset.datasets) == 1
            ), "Inference needs a single dataset per handler."
            if get_world_size() > 1 and dataset.datasets[0].has_sequences:
                sampler = TrackingInferenceSampler(
                    dataset.datasets[0]
                )  # pragma: no cover # pylint: disable=line-too-long
            elif get_world_size() > 1 and self.train_sampler is not None:
                # manually create distributed sampler for inference if using
                # custom training sampler
                sampler = DistributedSampler(  # pragma: no cover
                    dataset, shuffle=False
                )

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
