"""Detect data module."""
from typing import List, Optional

from projects.common.data_pipelines import CommonDataModule, default
from projects.common.datasets import (
    bdd100k_det_train,
    bdd100k_det_val,
    coco_train,
    coco_val,
)
from vis4d.data import BaseDatasetHandler, BaseSampleMapper, ScalabelDataset
from vis4d.data.transforms import BaseAugmentation, Resize


class DetectDataModule(CommonDataModule):
    """Detect data module."""

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        train_sample_mapper = BaseSampleMapper(
            category_map=self.category_mapping,
            data_backend=data_backend,
            skip_empty_samples=True,
        )
        test_sample_mapper = BaseSampleMapper(data_backend=data_backend)

        if self.experiment == "bdd100k":
            train_datasets = [
                ScalabelDataset(bdd100k_det_train(), True, train_sample_mapper)
            ]
            train_transforms = default((720, 1280))

            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(720, 1280))
            ]
            test_datasets = [
                ScalabelDataset(bdd100k_det_val(), False, test_sample_mapper)
            ]
        elif self.experiment == "coco":
            # train pipeline
            train_datasets = [
                ScalabelDataset(coco_train(), True, train_sample_mapper)
            ]
            train_transforms = default((800, 1333))

            # test pipeline
            test_transforms = [Resize(shape=(800, 133))]
            test_datasets = [
                ScalabelDataset(coco_val(), False, test_sample_mapper)
            ]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )

        train_handler = BaseDatasetHandler(
            train_datasets, transformations=train_transforms
        )
        test_handlers = [
            BaseDatasetHandler(
                ds, transformations=test_transforms, min_bboxes_area=0.0
            )
            for ds in test_datasets
        ]
        self.train_datasets = train_handler
        self.test_datasets = test_handlers
