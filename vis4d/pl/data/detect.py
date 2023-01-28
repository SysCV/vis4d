"""PyTorch Lightning detection module."""
from __future__ import annotations

from torch.utils.data import DataLoader

from vis4d.engine.data_to_revise.datasets import (
    coco_train,
    coco_val,
    coco_val_eval,
)
from vis4d.engine.data_to_revise.detect import (
    default_test_pipeline,
    default_train_pipeline,
)
from vis4d.eval import Evaluator

from .base import DataModule


class DetectDataModule(DataModule):
    """Detect data module."""

    def train_dataloader(self) -> DataLoader:
        """Setup training data pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        if self.experiment == "coco":
            dataloader = default_train_pipeline(
                coco_train(data_backend),
                self.samples_per_gpu,
                self.workers_per_gpu,
                (800, 1333),
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloader

    def test_dataloader(self) -> list[DataLoader]:
        """Setup inference pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        if self.experiment == "coco":
            dataloaders = default_test_pipeline(
                coco_val(data_backend),
                self.samples_per_gpu,
                self.workers_per_gpu,
                (800, 1333),
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloaders

    def evaluators(self) -> list[Evaluator]:
        """Define evaluators associated with test datasets."""
        if self.experiment == "bdd100k":
            raise NotImplementedError
        if self.experiment == "coco":
            evaluators = [coco_val_eval()]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return evaluators
