"""QD-3DT data module."""
from typing import List, Optional

from projects.common.data_pipelines import CommonDataModule, default
from projects.common.datasets import (
    kitti_det_train,
    kitti_track_map,
    kitti_track_train,
    kitti_track_val,
    nuscenes_mini_train,
    nuscenes_mini_val,
    nuscenes_track_map,
    nuscenes_train,
    nuscenes_val,
)
from vis4d.data import (
    BaseDatasetHandler,
    BaseReferenceSampler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.data.transforms import BaseAugmentation, Resize


class QD3DTDataModule(CommonDataModule):
    """QD-3DT data module."""

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        train_sample_mapper = BaseSampleMapper(
            data_backend=data_backend,
            inputs_to_load=("images", "intrinsics", "extrinsics"),
            targets_to_load=("boxes2d", "boxes3d"),
            skip_empty_samples=True,
        )
        test_sample_mapper = BaseSampleMapper(
            data_backend=data_backend,
            inputs_to_load=("images", "intrinsics", "extrinsics"),
        )

        if self.experiment == "kitti":
            # train pipeline
            train_sample_mapper.setup_categories(kitti_track_map)
            train_datasets = []
            train_datasets += [
                ScalabelDataset(
                    kitti_track_train(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(num_ref_imgs=1, scope=3),
                )
            ]
            train_datasets += [
                ScalabelDataset(
                    kitti_det_train(),
                    True,
                    train_sample_mapper,
                )
            ]
            train_transforms = default(im_hw=(375, 1242))

            # test pipeline
            test_sample_mapper.setup_categories(kitti_track_map)
            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(375, 1242))
            ]
            test_datasets = [
                ScalabelDataset(kitti_track_val(), False, test_sample_mapper)
            ]
        elif self.experiment == "nuscenes":
            # train pipeline
            train_sample_mapper.setup_categories(nuscenes_track_map)
            train_datasets = [
                ScalabelDataset(
                    nuscenes_train(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(scope=2, num_ref_imgs=1),
                )
            ]
            train_transforms = default(im_hw=(900, 1600))

            # test pipeline
            test_sample_mapper.setup_categories(nuscenes_track_map)
            test_transforms = [Resize(shape=(900, 1600))]
            test_datasets = [
                ScalabelDataset(nuscenes_val(), False, test_sample_mapper)
            ]
        elif self.experiment == "nuscenes_mini":
            # train pipeline
            train_sample_mapper.setup_categories(nuscenes_track_map)
            train_datasets = [
                ScalabelDataset(
                    nuscenes_mini_train(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(scope=2, num_ref_imgs=1),
                )
            ]
            train_transforms = default(im_hw=(900, 1600))

            # test pipeline
            test_sample_mapper.setup_categories(nuscenes_track_map)
            test_transforms = [Resize(shape=(900, 1600))]
            test_datasets = [
                ScalabelDataset(nuscenes_mini_val(), False, test_sample_mapper)
            ]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )

        train_handler = BaseDatasetHandler(
            train_datasets,
            transformations=train_transforms,
        )
        test_handlers = [
            BaseDatasetHandler(
                ds,
                transformations=test_transforms,
                clip_bboxes_to_image=False,
                min_bboxes_area=0.0,
            )
            for ds in test_datasets
        ]
        self.train_datasets = train_handler
        self.test_datasets = test_handlers
