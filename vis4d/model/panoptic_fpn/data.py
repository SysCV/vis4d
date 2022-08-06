"""PanopticFPN data module."""
from typing import List, Optional

from vis4d.common.data_pipelines import CommonDataModule, default
from vis4d.common.datasets import (
    bdd100k_pan_seg_train,
    bdd100k_pan_seg_val,
    bdd100k_seg_map,
    bdd100k_track_map,
)
from vis4d.data import BaseDatasetHandler, BaseSampleMapper, ScalabelDataset
from vis4d.data.transforms import BaseAugmentation, Resize


class PanopticFPNDataModule(CommonDataModule):
    """PanopticFPN data module."""

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        cat_map = {
            "boxes2d": bdd100k_track_map,
            "instance_masks": bdd100k_track_map,
            "semantic_masks": bdd100k_seg_map,
        }
        tgts = ("boxes2d", "instance_masks", "semantic_masks")
        train_sample_mapper = BaseSampleMapper(
            category_map=cat_map,
            targets_to_load=tgts,
            data_backend=data_backend,
            skip_empty_samples=True,
        )
        test_sample_mapper = BaseSampleMapper(
            category_map=cat_map,
            targets_to_load=tgts,
            data_backend=data_backend,
        )

        train_datasets, train_transforms = [], None
        if self.experiment == "bdd100k":
            if stage is None or stage == "fit":
                train_datasets += [
                    ScalabelDataset(
                        bdd100k_pan_seg_train(), True, train_sample_mapper
                    )
                ]
                train_transforms = default((720, 1280))

            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(720, 1280))
            ]
            test_datasets = [
                ScalabelDataset(
                    bdd100k_pan_seg_val(), False, test_sample_mapper
                )
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
