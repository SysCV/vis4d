"""QDTrack data module."""
from typing import List, Optional

from projects.common.data_pipelines import default, mosaic_mixup
from projects.common.datasets import (
    bdd100k_det_train,
    bdd100k_track_train,
    bdd100k_track_val,
    crowdhuman_trainval,
    mot17_train,
    mot17_val,
    mot20_train,
    mot20_val,
)
from vis4d.common.io import BaseDataBackend, FileBackend, HDF5Backend
from vis4d.data import (
    BaseDatasetHandler,
    BaseReferenceSampler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.data.module import BaseDataModule
from vis4d.data.transforms import BaseAugmentation, Resize
from vis4d.struct import ArgsType


class QDTrackDataModule(BaseDataModule):
    """QDTrack data module."""

    def __init__(
        self,
        experiment: str,
        *args: ArgsType,
        use_hdf5: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.experiment = experiment
        self.use_hdf5 = use_hdf5

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        if not self.use_hdf5:
            data_backend: BaseDataBackend = FileBackend()
        else:
            data_backend = HDF5Backend()

        train_sample_mapper = BaseSampleMapper(
            category_map=self.category_mapping,
            data_backend=data_backend,
            skip_empty_samples=True,
        )
        test_sample_mapper = BaseSampleMapper(data_backend=data_backend)

        clip_to_image = True
        if self.experiment == "mot17":
            # train pipeline
            train_datasets = []
            train_datasets += [
                ScalabelDataset(
                    crowdhuman_trainval(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(num_ref_imgs=1, scope=0),
                )
            ]
            train_datasets += [
                ScalabelDataset(
                    mot17_train(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(
                        scope=10, num_ref_imgs=1, skip_nomatch_samples=True
                    ),
                )
            ]
            train_transforms = mosaic_mixup(
                (800, 1440), clip_inside_image=False
            )
            clip_to_image = False

            # test pipeline
            test_transforms: List[BaseAugmentation] = [
                Resize(shape=(800, 1440))
            ]
            test_datasets = [
                ScalabelDataset(mot17_val(), False, test_sample_mapper)
            ]
        elif self.experiment == "mot20":
            # train pipeline
            train_datasets = []
            train_datasets += [
                ScalabelDataset(
                    crowdhuman_trainval(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(num_ref_imgs=1, scope=0),
                )
            ]
            train_datasets += [
                ScalabelDataset(
                    mot20_train(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(
                        scope=10, num_ref_imgs=1, skip_nomatch_samples=True
                    ),
                )
            ]
            train_transforms = mosaic_mixup(
                (896, 1600),
                multiscale_sizes=[(32 * i, 1600) for i in range(20, 36)],
            )

            # test pipeline
            test_transforms = [Resize(shape=(896, 1600))]
            test_datasets = [
                ScalabelDataset(mot20_val(), False, test_sample_mapper)
            ]
        elif self.experiment == "bdd100k":
            train_datasets = []
            train_datasets += [
                ScalabelDataset(
                    bdd100k_det_train(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(num_ref_imgs=1, scope=0),
                )
            ]
            train_datasets += [
                ScalabelDataset(
                    bdd100k_track_train(),
                    True,
                    train_sample_mapper,
                    BaseReferenceSampler(
                        scope=10, num_ref_imgs=1, skip_nomatch_samples=True
                    ),
                )
            ]

            train_transforms = default((720, 1280))

            test_transforms = [Resize(shape=(720, 1280))]
            test_datasets = [
                ScalabelDataset(bdd100k_track_val(), False, test_sample_mapper)
            ]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )

        train_handler = BaseDatasetHandler(
            train_datasets,
            clip_bboxes_to_image=clip_to_image,
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
