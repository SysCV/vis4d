"""Segment data module.

TODO(fyu) to find a proper place for this code.
"""
from typing import List, Optional, Tuple

from vis4d.common_to_clean.data_pipelines import (
    CommonDataModule,
    add_colorjitter,
)
from vis4d.common_to_clean.datasets import (
    bdd100k_sem_seg_train,
    bdd100k_sem_seg_val,
)
from vis4d.data_to_clean import (
    BaseDatasetHandler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.data_to_clean.transforms import (
    BaseAugmentation,
    KorniaRandomHorizontalFlip,
    RandomCrop,
    Resize,
)


def multi_scale(
    im_hw: Tuple[int, int], crop_hw: Tuple[int, int]
) -> List[BaseAugmentation]:
    """Generate multi-scale training augmentation pipeline."""
    augs: List[BaseAugmentation] = []
    augs += [Resize(shape=im_hw, scale_range=(0.5, 2.0), keep_ratio=True)]
    augs += [
        RandomCrop(shape=crop_hw, crop_type="absolute", cat_max_ratio=0.75)
    ]
    augs += [KorniaRandomHorizontalFlip(prob=0.5)]
    return augs


class SegmentDataModule(CommonDataModule):
    """Segment data module."""

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        train_sample_mapper = BaseSampleMapper(
            category_map=self.category_mapping,
            targets_to_load=("semantic_masks",),
            data_backend=data_backend,
            skip_empty_samples=True,
        )
        test_sample_mapper = BaseSampleMapper(
            category_map=self.category_mapping,
            targets_to_load=("semantic_masks",),
            data_backend=data_backend,
        )

        if self.experiment == "bdd100k":
            train_datasets = [
                ScalabelDataset(
                    bdd100k_sem_seg_train(), True, train_sample_mapper
                )
            ]
            train_transforms = multi_scale((720, 1280), (512, 1024))
            add_colorjitter(train_transforms, p=1.0)

            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(720, 1280))
            ]
            test_datasets = [
                ScalabelDataset(
                    bdd100k_sem_seg_val(), False, test_sample_mapper
                )
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
