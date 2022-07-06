"""
    This file contains the implementation of the datamodule that is needed
    to run the deep sort project.
    It creates different datasets (train, valid, test) for the different
    stages and applies augmentations.
"""
from typing import List, Optional

from projects.common.data_pipelines import CommonDataModule, default
from projects.common.datasets import (
    bdd100k_det_map,
    bdd100k_det_train,
    bdd100k_det_val,
)
from vis4d.data import BaseDatasetHandler, BaseSampleMapper, ScalabelDataset
from vis4d.data.transforms import BaseAugmentation, Resize


class DetectDataModule(CommonDataModule):
    """
    Data module that loads detection datsets.
    Note, even though deepsort is a tracking framework it is trained using
    detection data (i.e. the similarity head)
    """

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        # Scalabel dependencies
        train_sample_mapper = BaseSampleMapper(
            data_backend=data_backend,
            skip_empty_samples=True,
        )
        test_sample_mapper = BaseSampleMapper(data_backend=data_backend)

        train_datasets, train_transforms = [], None
        if self.experiment == "bdd100k":
            # Train datset setup
            if stage is None or stage == "fit":
                train_sample_mapper.setup_categories(bdd100k_det_map)
                train_datasets = [
                    ScalabelDataset(
                        bdd100k_det_train(), True, train_sample_mapper
                    )
                ]
                # Default image augmentations
                train_transforms = default((720, 1280))

            # Test dataset setup
            test_sample_mapper.setup_categories(bdd100k_det_map)
            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(720, 1280))
            ]
            test_datasets = [
                ScalabelDataset(bdd100k_det_val(), False, test_sample_mapper)
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
