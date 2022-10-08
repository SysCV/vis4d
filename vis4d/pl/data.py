"""Data module composing the data loading pipeline."""
from typing import List, Optional, Union, no_type_check

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_info
from torch.utils import data

from vis4d.data.io import DataBackend, FileBackend, HDF5Backend
from vis4d.eval.base import Evaluator
from vis4d.pl.callbacks.evaluator import DefaultEvaluatorCallback
from vis4d.pl.callbacks.writer import DefaultWriterCallback


class DataModule(pl.LightningDataModule):
    """Default data module for Vis4D.

    Attributes:
        experiment: Experiment string to ensemble multiple configurations in
        single data module
        samples_per_gpu: batch size per GPU.
        workers_per_gpu: dataloader workers per GPU.
        pin_memory: Whether to allocate specific GPU memory for the dataloader
        workers.
        use_hdf5: Whether to use hdf5 data backend.
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
        experiment: Optional[str] = None,
        samples_per_gpu: int = 1,
        workers_per_gpu: int = 1,
        pin_memory: bool = False,
        use_hdf5: bool = False,
        visualize: bool = False,
        input_dir: Optional[str] = None,
        video_based_inference: Optional[bool] = None,
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        self.experiment = experiment
        self.visualize = visualize
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.pin_memory = pin_memory
        self.use_hdf5 = use_hdf5
        self.input_dir = input_dir
        self.video_based_inference = video_based_inference

    def _setup_backend(self) -> DataBackend:
        """Setup data backend."""
        backend = FileBackend() if not self.use_hdf5 else HDF5Backend()
        rank_zero_info("Using data backend: %s", backend.__class__.__name__)
        return backend

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

        # TODO is this still necessary?
        # if self._sampler_cfg is not None:
        #     self.trainer._accelerator_connector.replace_sampler_ddp = False

    def setup_data_callbacks(self, stage: str, log_dir: str) -> List[Callback]:
        """Setup callbacks for evaluation and prediction writing."""
        if stage == "predict":
            return self.writers()  # TODO Implement
        else:
            return [
                DefaultEvaluatorCallback(i, ev)
                for i, ev in enumerate(self.evaluators())
            ]

    def train_dataloader(self) -> data.DataLoader:
        """Return dataloader for training."""
        raise NotImplementedError

    def evaluators(self) -> List[Evaluator]:
        """Define Evaluators used in test stage."""
        return []

    # def writers(self) -> List[BaseWriter]:
    #     """Define writers used in predict stage."""
    #     return []

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
