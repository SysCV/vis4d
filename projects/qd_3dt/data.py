"""QD-3DT data module."""
from typing import Optional

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
from vis4d.data.transforms import Resize


class QD3DTDataModule(CommonDataModule):
    """QD-3DT data module."""

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        data_backend = self._setup_backend()

        if self.experiment == "kitti":
            category_mapping = kitti_track_map
            train_dataset_list = [kitti_track_train, kitti_det_train]
            ref_sampler_list = [
                BaseReferenceSampler(scope=3, num_ref_imgs=1),
                None,
            ]
            test_dataset_list = [kitti_track_val]

            train_transforms = default(im_hw=(375, 1242))
            test_transforms = [Resize(shape=(375, 1242))]
        elif "nuscenes" in self.experiment:
            category_mapping = nuscenes_track_map
            if self.experiment == "nuscenes_mini":
                train_dataset_list = [nuscenes_mini_train]
                test_dataset_list = [nuscenes_mini_val]
            else:
                train_dataset_list = [nuscenes_train]
                test_dataset_list = [nuscenes_val]
            ref_sampler_list = [BaseReferenceSampler(scope=2, num_ref_imgs=1)]

            train_transforms = default(im_hw=(900, 1600))
            test_transforms = [Resize(shape=(900, 1600))]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )

        # train pipeline
        train_datasets = []
        if stage is None or stage == "fit":
            train_sample_mapper = BaseSampleMapper(
                data_backend=data_backend,
                inputs_to_load=("images", "intrinsics", "extrinsics"),
                targets_to_load=("boxes2d", "boxes3d"),
                skip_empty_samples=True,
                category_map=category_mapping,
            )

            for dataset, ref_sampler in zip(
                train_dataset_list, ref_sampler_list
            ):
                train_datasets.append(
                    ScalabelDataset(
                        dataset(),
                        training=True,
                        mapper=train_sample_mapper,
                        ref_sampler=ref_sampler,
                    )
                )
            self.train_datasets = BaseDatasetHandler(
                train_datasets,
                transformations=train_transforms,
            )

        # test pipeline
        self.test_datasets = []
        test_sample_mapper = BaseSampleMapper(
            data_backend=data_backend,
            inputs_to_load=("images", "intrinsics", "extrinsics"),
            category_map=category_mapping,
        )

        for dataset in test_dataset_list:
            test_dataset = ScalabelDataset(
                dataset(), training=False, mapper=test_sample_mapper
            )
            self.test_datasets.append(
                BaseDatasetHandler(
                    test_dataset,
                    transformations=test_transforms,
                    clip_bboxes_to_image=False,
                    min_bboxes_area=0.0,
                )
            )
