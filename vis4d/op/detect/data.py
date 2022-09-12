"""Detect data module."""
from typing import List, Optional

from vis4d.common_to_revise.data_pipelines import CommonDataModule, default
from vis4d.common_to_revise.datasets import (
    bdd100k_det_map,
    bdd100k_det_train,
    bdd100k_det_val,
    coco_det_map,
    coco_train,
    coco_val,
)
from vis4d.data_to_revise import (
    BaseDatasetHandler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.data_to_revise.transforms import BaseAugmentation, Resize


class DetectDataModule(CommonDataModule):
    """Detect data module."""

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        train_sample_mapper = BaseSampleMapper(
            data_backend=data_backend,
            skip_empty_samples=True,
        )
        test_sample_mapper = BaseSampleMapper(data_backend=data_backend)

        train_datasets, train_transforms = [], None
        if self.experiment == "bdd100k":
            if stage is None or stage == "fit":
                train_sample_mapper.setup_categories(bdd100k_det_map)
                train_datasets = [
                    ScalabelDataset(
                        bdd100k_det_train(), True, train_sample_mapper
                    )
                ]
                train_transforms = default((720, 1280))

            test_sample_mapper.setup_categories(bdd100k_det_map)
            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(720, 1280))
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
            test_transforms = [Resize(shape=(800, 1333))]
            test_datasets = [
                ScalabelDataset(coco_val(), False, test_sample_mapper)
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
