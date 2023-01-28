"""PyTorch Lightning detection module."""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from vis4d.data.datasets import COCO
from vis4d.data.io.base import DataBackend
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose, random_apply
from vis4d.data.transforms.flip import flip_boxes2d, flip_image
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.data.transforms.resize import (
    resize_boxes2d,
    resize_image,
    resize_masks,
)
from vis4d.eval import COCOEvaluator, Evaluator

from .base import DataModule


# TODO MOVE TO CONFIG
def coco_train(data_backend: DataBackend) -> COCO:
    """Create COCO train dataset with default data root."""
    return COCO("data/COCO/", data_backend=data_backend)


def coco_val(data_backend: DataBackend) -> COCO:
    """Create COCO val dataset with default data root."""
    return COCO("data/COCO/", split="val2017", data_backend=data_backend)


def coco_val_eval() -> COCOEvaluator:
    """Create COCO val evaluator with default data root."""
    return COCOEvaluator("data/COCO/", split="val2017")


def default_train_pipeline(
    datasets: Dataset | list[Dataset],
    batch_size: int,
    num_workers: int,
    im_hw: tuple[int, int],
    with_mask: bool = False,
) -> DataLoader:
    """Default train preprocessing pipeline for detectors."""
    resize_trans = [resize_image(im_hw, keep_ratio=True), resize_boxes2d()]
    flip_trans = [flip_image(), flip_boxes2d()]
    if with_mask:
        resize_trans += [resize_masks()]
        flip_trans += [flip_image()]
    preprocess_fn = compose(
        [*resize_trans, random_apply(flip_trans), normalize_image()]
    )
    batchprocess_fn = pad_image()
    datapipe = DataPipe(datasets, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        batchprocess_fn=batchprocess_fn,
    )
    return train_loader


def default_test_pipeline(
    datasets: Dataset | list[Dataset],
    batch_size: int,
    num_workers: int,
    im_hw: tuple[int, int],
) -> list[DataLoader]:
    """Default test preprocessing pipeline for detectors."""
    preprocess_fn = compose(
        [
            resize_image(im_hw, keep_ratio=True, align_long_edge=True),
            normalize_image(),
        ]
    )
    batchprocess_fn = pad_image()
    datapipe = DataPipe(datasets, preprocess_fn)
    test_loaders = build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        batchprocess_fn=batchprocess_fn,
    )
    return test_loaders


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
