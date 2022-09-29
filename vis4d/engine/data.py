"""Data module composing the data loading pipeline."""
import itertools
import os.path as osp
from typing import List, Optional, Union, no_type_check

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils import data

from vis4d.engine.callbacks.evaluator import DefaultEvaluatorCallback
from vis4d.engine.callbacks.writer import DefaultWriterCallback


def build_callbacks(
    datasets,
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


class BaseDataModule(pl.LightningDataModule):
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
        raise NotImplementedError

    def predict_dataloader(
        self,
    ) -> Union[data.DataLoader, List[data.DataLoader]]:
        """Return dataloaders for prediction."""
        return self.test_dataloader()  # pragma: no cover

    def val_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for validation."""
        return self.test_dataloader()

    def test_dataloader(self) -> List[data.DataLoader]:
        """Return dataloaders for testing."""
        raise NotImplementedError
