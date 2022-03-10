"""Data module composing the data loading pipeline."""
import itertools
import os.path as osp
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from ..common.registry import RegistryHolder
from ..common.utils import get_world_size
from ..config import Config
from ..engine.evaluator import StandardEvaluatorCallback
from ..struct import InputSample
from .dataset import ScalabelDataset
from .datasets import BaseDatasetLoader
from .handler import BaseDatasetHandler
from .samplers import BaseSampler, TrackingInferenceSampler
from .transforms import BaseAugmentation
from .utils import identity_batch_collator


def build_callbacks(
    datasets: List[ScalabelDataset],
    out_dir: Optional[str] = None,
    is_predict: bool = False,
    visualize: bool = False,
) -> List[Callback]:
    """Build callbacks."""
    callbacks: List[Callback] = []
    for i, d in enumerate(datasets):
        out = (
            osp.join(out_dir, d.dataset.name) if out_dir is not None else None
        )
        if not is_predict:
            callbacks.append(StandardEvaluatorCallback(i, d.dataset, out))
        else:
            assert out is not None
            callbacks.append(ScalabelWriterCallback(i, out, visualize))
    return callbacks


class BaseDataModule(pl.LightningDataModule, metaclass=RegistryHolder):
    """Default data module for Vis4D."""

    def __init__(
        self,
        samples_per_gpu: int = 1,
        workers_per_gpu: int = 1,
        pin_memory: bool = False,
        visualize: bool = False,
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        self.visualize = visualize
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.pin_memory = pin_memory
        self.train_datasets: Optional[BaseDatasetHandler] = None
        self.test_datasets: Optional[List[BaseDatasetHandler]] = None
        self.predict_datasets: Optional[List[BaseDatasetHandler]] = None
        self.train_sampler: Optional[BaseSampler] = None

    def prepare_data(self):
        """Data preparation operations to perform on the master process.

        Do things that might write to disk or that need to be done only from
        a single process in distributed settings.
        """

    def setup(self, stage: Optional[str] = None):
        """Data preparation operations to perform on every GPU.

        Setup:
        - Set Train / Test / Predict Datasets
        - Wrap datasets into handlers with transformations
        - Setup data callbacks
        - Optionally: Define train sampler for custom dataset sampling.
        """
        raise NotImplementedError

    def convert_input_dir_to_dataset(self):
        if cfg.launch.input_dir:  # TODO needs refinement
            input_dir = cfg.launch.input_dir
            if input_dir[-1] == "/":
                input_dir = input_dir[:-1]
            dataset_name = osp.basename(input_dir)
            self.predict_datasets = [
                Custom(name=dataset_name, data_root=input_dir)
            ]
        else:
            self.predict_datasets = self.test_datasets

    def setup_data_callbacks(self, stage: str, log_dir: str) -> List[Callback]:
        """setup callbacks"""
        if stage == "fit":
            if self.test_datasets is not None and len(self.test_datasets) > 0:
                # TODO callbacks vs dataset handlers in inference sync
                test_datasets = list(
                    itertools.chain(*[h.datasets for h in self.test_datasets])
                )
                return build_callbacks(test_datasets)
        elif stage == "test":
            assert self.test_datasets is not None and len(
                self.test_datasets
            ), "No test datasets specified!"
            test_datasets = list(
                itertools.chain(*[h.datasets for h in self.test_datasets])
            )
            return build_callbacks(test_datasets, log_dir)
        elif stage == "predict":
            assert (
                self.predict_datasets is not None
                and len(self.predict_datasets) > 0
            ), "No predict datasets specified!"
            predict_datasets = list(
                itertools.chain(*[h.datasets for h in self.predict_datasets])
            )
            return build_callbacks(
                predict_datasets, log_dir, True, self.visualize
            )
        elif stage == "tune":
            assert self.test_datasets is not None and len(
                self.test_datasets
            ), "No test datasets specified!"
            test_datasets = list(
                itertools.chain(*[h.datasets for h in self.test_datasets])
            )
            return build_callbacks(test_datasets, log_dir)
        else:
            raise NotImplementedError(f"Action {stage} not known!")

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
