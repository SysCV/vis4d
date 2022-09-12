"""Data module composing the data loading pipeline."""
import itertools
import os.path as osp
from typing import List, Optional, Union, no_type_check

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from vis4d.data_to_revise.callbacks.evaluator import DefaultEvaluatorCallback
from vis4d.data_to_revise.callbacks.writer import DefaultWriterCallback

from ..common_to_revise.registry import RegistryHolder
from ..common_to_revise.utils import get_world_size
from ..struct_to_revise import CategoryMap, InputSample, ModuleCfg
from .dataset import ScalabelDataset
from .datasets import Custom
from .handler import BaseDatasetHandler
from .samplers import TrackingInferenceSampler, build_data_sampler
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
            callbacks.append(DefaultEvaluatorCallback(i, d.dataset, out))
        else:
            assert out is not None
            callbacks.append(
                DefaultWriterCallback(i, d.dataset, out, visualize)
            )
    return callbacks


class BaseDataModule(pl.LightningDataModule, metaclass=RegistryHolder):
    """Default data module for Vis4D.

    Attributes:
        samples_per_gpu: batch size per GPU.
        workers_per_gpu: dataloader workers per GPU.
        pin_memory: Whether to allocate specific GPU memory for the dataloader
        workers.
        visualize: If you're running in predict mode, this option lets you
        visualize the model predictions in the output_dir.
        input_dir: Directory in case you want to run inference on a folder
        with input data (e.g. images that can be temporally sorted by name).
        video_based_inference: If the dataset contains videos, the default
        inference mode will distribute the videos to the GPUs instead of
        individual frames, s.t. a tracking algorithm can work correctly.
        This option allows to modify the default behavior (i.e. turning off
        video inference) if this is not desired.
    """

    def __init__(
        self,
        samples_per_gpu: int = 1,
        workers_per_gpu: int = 1,
        pin_memory: bool = False,
        visualize: bool = False,
        input_dir: Optional[str] = None,
        sampler_cfg: Optional[ModuleCfg] = None,
        video_based_inference: Optional[bool] = None,
        subcommand: Optional[str] = None,
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        self.visualize = visualize
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.pin_memory = pin_memory
        self.input_dir = input_dir
        self.video_based_inference = video_based_inference
        self.train_datasets: Optional[BaseDatasetHandler] = None
        self.test_datasets: Optional[List[BaseDatasetHandler]] = None
        self.predict_datasets: Optional[List[BaseDatasetHandler]] = None
        self._sampler_cfg = sampler_cfg
        self.category_mapping: Optional[CategoryMap] = None
        self.create_datasets(subcommand)

    def set_category_mapping(self, cat_map: CategoryMap) -> None:
        """Set default category mapping used when creating the datasets."""
        self.category_mapping = cat_map

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Create Train / Test / Predict Datasets."""
        raise NotImplementedError

    @no_type_check
    def setup(self, stage: Optional[str] = None) -> None:
        """Data preparation operations to perform on every GPU.

        - Setup data callbacks
        - Set sampler for DDP
        """
        self.trainer.callbacks += self.setup_data_callbacks(
            stage, self.trainer.log_dir
        )
        # pylint: disable=protected-access
        self.trainer._callback_connector._attach_model_logging_functions()
        self.trainer.callbacks = (
            self.trainer._callback_connector._reorder_callbacks(
                self.trainer.callbacks
            )
        )
        if self._sampler_cfg is not None:
            self.trainer._accelerator_connector.replace_sampler_ddp = False

    def convert_input_dir_to_dataset(self) -> None:
        """Convert a given input directory to a dataset for prediction."""
        if self.input_dir is not None:
            if self.input_dir is not None:
                if not osp.exists(self.input_dir):
                    raise FileNotFoundError(
                        f"Input directory does not exist: {self.input_dir}"
                    )
            if self.input_dir[-1] == "/":
                self.input_dir = self.input_dir[:-1]
            dataset_name = osp.basename(self.input_dir)
            dataset = [
                ScalabelDataset(Custom(dataset_name, self.input_dir), False)
            ]
            self.predict_datasets = [BaseDatasetHandler(dataset)]
        else:
            self.predict_datasets = self.test_datasets  # pragma: no cover

    def setup_data_callbacks(self, stage: str, log_dir: str) -> List[Callback]:
        """Setup callbacks for evaluation and prediction writing."""
        if stage == "fit":
            if self.test_datasets is not None and len(self.test_datasets) > 0:
                test_datasets = list(
                    itertools.chain(*[h.datasets for h in self.test_datasets])
                )
                return build_callbacks(test_datasets)
            return []  # pragma: no cover
        if stage in ["test", "tune"]:
            assert self.test_datasets is not None and len(
                self.test_datasets
            ), "No test datasets specified!"
            test_datasets = list(
                itertools.chain(*[h.datasets for h in self.test_datasets])
            )
            return build_callbacks(test_datasets, log_dir)
        if stage == "predict":
            self.convert_input_dir_to_dataset()
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
        raise NotImplementedError(f"Action {stage} not known!")

    def train_dataloader(self) -> data.DataLoader:
        """Return dataloader for training."""
        assert self.train_datasets is not None, "No train datasets specified!"
        if self._sampler_cfg is not None:
            train_sampler = build_data_sampler(
                self._sampler_cfg, self.train_datasets, self.samples_per_gpu
            )
            batch_size, shuffle = 1, False
        else:
            batch_size, shuffle, train_sampler = (
                self.samples_per_gpu,
                True,
                None,
            )
        train_dataloader = data.DataLoader(
            self.train_datasets,
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
        self, datasets: List[BaseDatasetHandler]
    ) -> List[data.DataLoader]:
        """Build dataloaders for test / predict."""
        dataloaders = []
        for dataset in datasets:
            sampler: Optional[data.Sampler] = None
            assert (
                len(dataset.datasets) == 1
            ), "Inference needs a single dataset per handler."

            current_dataset = dataset.datasets[0]
            video_based_inference = (
                False
                if self.video_based_inference is None
                else self.video_based_inference
            )
            if isinstance(current_dataset, ScalabelDataset):
                if not dataset.datasets[0].has_sequences:
                    video_based_inference = False
                elif self.video_based_inference is None:
                    video_based_inference = True

            if (
                get_world_size() > 1 and video_based_inference
            ):  # pragma: no cover
                assert isinstance(current_dataset, ScalabelDataset), (
                    "Need type ScalabelDataset for TrackingInferenceSampler"
                    " to split dataset by sequences!"
                )
                sampler = TrackingInferenceSampler(current_dataset)
            elif (
                get_world_size() > 1 and self._sampler_cfg is not None
            ):  # pragma: no cover
                # manually create distributed sampler for inference if using
                # custom training sampler
                sampler = DistributedSampler(dataset, shuffle=False)

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
