"""Detect data module."""
from typing import List, Tuple, Union

from torch.utils.data import DataLoader, Dataset

from vis4d.common_to_revise.datasets import (
    bdd100k_det_map,
    bdd100k_det_train,
    bdd100k_det_val,
    bdd100k_track_map,
)
from vis4d.data.datasets import COCO
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose, random_apply
from vis4d.data.transforms.flip import boxes2d_flip, image_flip
from vis4d.data.transforms.normalize import image_normalize
from vis4d.data.transforms.pad import image_pad
from vis4d.data.transforms.resize import resize_boxes2d, resize_image
from vis4d.eval import COCOEvaluator, Evaluator
from vis4d.pl.data import DataModule


class DetectDataModule(DataModule):
    """Detect data module."""

    def _build_train_dataloder(
        self,
        datasets: Union[Dataset, List[Dataset]],
        batch_size: int,
        im_hw: Tuple[int, int],
    ) -> DataLoader:
        """Build dataloader incl. data preprocessing pipeline."""
        preprocess_fn = compose(
            [
                resize_image(im_hw, keep_ratio=True),
                resize_boxes2d(),
                random_apply(
                    [
                        image_flip(),
                        boxes2d_flip(),
                    ]
                ),
                image_normalize(),
            ]
        )
        batchprocess_fn = image_pad()

        datapipe = DataPipe(datasets, preprocess_fn)
        train_loader = build_train_dataloader(
            datapipe,
            samples_per_gpu=batch_size,
            batchprocess_fn=batchprocess_fn,
        )
        return train_loader

    def train_dataloader(self) -> DataLoader:
        """Setup training data pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            dataset = COCO("data/COCO/", data_backend=data_backend)
            dataloader = self._build_train_dataloder(
                dataset, self.samples_per_gpu, (800, 1333)
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloader

    def _build_test_dataloaders(
        self,
        datasets: Union[Dataset, List[Dataset]],
        batch_size: int,
        im_hw: Tuple[int, int],
    ) -> List[DataLoader]:
        """Generate default test data pipeline."""
        preprocess_fn = compose(
            [
                resize_image(im_hw, keep_ratio=True, align_long_edge=True),
                image_normalize(),
            ]
        )
        batchprocess_fn = image_pad()

        datapipe = DataPipe(datasets, preprocess_fn)
        test_loaders = build_inference_dataloaders(
            datapipe,
            samples_per_gpu=batch_size,
            batchprocess_fn=batchprocess_fn,
        )
        return test_loaders

    def test_dataloader(self) -> List[DataLoader]:
        """Setup inference pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            dataset = COCO(
                "data/COCO/", data_backend=data_backend, split="val2017"
            )
            dataloaders = self._build_test_dataloaders(
                dataset, self.samples_per_gpu, (800, 1333)
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloaders

    def evaluators(self) -> List[Evaluator]:
        """Define evaluators associated with test datasets."""
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            evaluators = [COCOEvaluator("data/COCO/", split="val2017")]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return evaluators


class InsSegDataModule(DataModule):
    """InsSeg data module."""

    def train_dataloader(self) -> DataLoader:
        """Setup training data pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            dataset = COCO(
                "data/COCO/", with_mask=True, data_backend=data_backend
            )
            dataloader = default_train(
                dataset, self.samples_per_gpu, (800, 1333)
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloader

    def test_dataloader(self) -> List[DataLoader]:
        """Setup inference pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            dataset = COCO(
                "data/COCO/",
                with_mask=True,
                split="val2017",
                data_backend=data_backend,
            )
            dataloaders = default_test(
                dataset, self.samples_per_gpu, (800, 1333)
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloaders

    def evaluators(self) -> List[Evaluator]:
        """Define evaluators associated with test datasets."""
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            evaluators = [
                COCOEvaluator("data/COCO/", split="val2017"),
                COCOEvaluator("data/COCO/", iou_type="segm", split="val2017"),
            ]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return evaluators
