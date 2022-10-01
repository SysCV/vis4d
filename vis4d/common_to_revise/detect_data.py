"""Detect data module."""
from typing import List, Optional

from torch.utils.data import DataLoader

from vis4d.common_to_revise.data_pipelines import (
    CommonDataModule,
    default_test,
    default_train,
)
from vis4d.common_to_revise.datasets import (
    bdd100k_det_map,
    bdd100k_det_train,
    bdd100k_det_val,
    bdd100k_track_map,
    coco_insseg_val,
    coco_train,
    coco_val,
)
from vis4d.data.datasets import COCO
from vis4d.eval import BaseEvaluator, COCOEvaluator


class DetectDataModule(CommonDataModule):
    """Detect data module."""

    def train_dataloader(self) -> DataLoader:
        """Setup training data pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            dataset = COCO("data/COCO/", data_backend=data_backend)
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
                "data/COCO/", data_backend=data_backend, split="val2017"
            )
            dataloaders = default_test(
                dataset, self.samples_per_gpu, (800, 1333)
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloaders

    def evaluators(self) -> List[BaseEvaluator]:
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


# TODO revise
class InsSegDataModule(CommonDataModule):
    """InsSeg data module."""

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        train_sample_mapper = BaseSampleMapper(
            data_backend=data_backend,
            skip_empty_samples=True,
            targets_to_load=("boxes2d", "instance_masks"),
        )
        test_sample_mapper = BaseSampleMapper(data_backend=data_backend)

        train_datasets, train_transforms = [], None
        if self.experiment == "bdd100k":
            if stage is None or stage == "fit":
                train_sample_mapper.setup_categories(bdd100k_track_map)
                train_datasets = [
                    ScalabelDataset(
                        bdd100k_det_train(), True, train_sample_mapper
                    )
                ]
                train_transforms = default((720, 1280))

            test_sample_mapper.setup_categories(bdd100k_track_map)
            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(720, 1280), keep_ratio=True)
            ]
            test_datasets = [
                ScalabelDataset(bdd100k_det_val(), False, test_sample_mapper)
            ]
        elif self.experiment == "coco":
            if stage is None or stage == "fit":
                train_sample_mapper.setup_categories(coco_det_map)
                train_datasets = [
                    ScalabelDataset(coco_train(), True, train_sample_mapper)
                ]
                train_transforms = default((800, 1333))

            test_sample_mapper.setup_categories(coco_det_map)
            test_transforms = [
                Resize(
                    shape=(800, 1333), keep_ratio=True, align_long_edge=True
                )
            ]
            test_datasets = [
                ScalabelDataset(coco_insseg_val(), False, test_sample_mapper)
            ]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )

        if len(train_datasets) > 0:
            train_handler = BaseDatasetHandler(
                train_datasets, transformations=train_transforms
            )
            self.train_datasets = train_handler

        test_handlers = [
            BaseDatasetHandler(
                ds, transformations=test_transforms, min_bboxes_area=0.0
            )
            for ds in test_datasets
        ]
        self.test_datasets = test_handlers
