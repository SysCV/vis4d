"""Cmd line tool for QDTrack."""
import argparse
from typing import List, Optional, Tuple

from projects.common.data_pipelines import default, mosaic_mixup, multi_scale
from projects.common.datasets import (
    bdd100k_det_train,
    bdd100k_track_map,
    bdd100k_track_train,
    bdd100k_track_val,
    crowdhuman_trainval,
    mot17_map,
    mot17_train,
    mot17_val,
)
from projects.common.models import build_faster_rcnn, build_yolox
from projects.qdtrack.qdtrack import QDTrack, QDTrackYOLOX
from vis4d.common.bbox.poolers import MultiScaleRoIAlign
from vis4d.common.io import FileBackend, HDF5Backend
from vis4d.config.defaults import default_argument_parser
from vis4d.data import (
    BaseDatasetHandler,
    BaseReferenceSampler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.data.module import BaseDataModule
from vis4d.data.transforms import Resize
from vis4d.engine.trainer import BaseCLI, DefaultTrainer
from vis4d.model.track.graph import QDTrackGraph
from vis4d.model.track.similarity import QDSimilarityHead
from vis4d.struct import CategoryMap


def setup_model(experiment: str, detector: str = "") -> QDTrack:
    """Setup model with experiment specific hyperparameters"""
    if experiment == "mot17":
        if detector == "YOLOX":
            track_graph = QDTrackGraph(
                keep_in_memory=30, init_score_thr=0.7, obj_score_thr=0.15
            )
        else:
            track_graph = QDTrackGraph(
                keep_in_memory=30, init_score_thr=0.9, obj_score_thr=0.5
            )
        category_mapping = mot17_map
    elif experiment == "bdd100k":
        track_graph = QDTrackGraph(keep_in_memory=10)
        category_mapping = bdd100k_track_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    if detector == "YOLOX":
        detector = build_yolox(category_mapping)
        similarity_head = QDSimilarityHead(
            proposal_pooler=MultiScaleRoIAlign(0, (7, 7), [8, 16, 32])
        )
        model = QDTrackYOLOX(
            detection=detector,
            similarity=similarity_head,
            track_graph=track_graph,
        )
    else:
        detector = build_faster_rcnn(category_mapping)
        similarity_head = QDSimilarityHead()
        model = QDTrack(
            category_mapping=category_mapping,
            detection=detector,
            similarity=similarity_head,
            track_graph=track_graph,
        )

    if experiment == "mot17":
        model.detector.clip_bboxes_to_image = False
    return model


class QDTrackDataModule(BaseDataModule):
    def __init__(
        self, experiment: str, use_hdf5: bool = False, *args, **kwargs
    ) -> None:
        """"""
        super().__init__(*args, **kwargs)
        self.experiment = experiment
        self.use_hdf5 = use_hdf5

    def set_cat_map(self, cat_map: CategoryMap) -> None:
        self.cat_map = cat_map

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup data pipelines for each experiment."""
        if not self.use_hdf5:
            data_backend = FileBackend()
        else:
            data_backend = HDF5Backend()

        train_sample_mapper = BaseSampleMapper(
            category_mapping=self.cat_map,
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
            test_transforms = [Resize(shape=(800, 1440))]
            test_datasets = [
                ScalabelDataset(mot17_val(), False, test_sample_mapper)
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
        # TODO input dir hanlding
        self.trainer.callbacks += self.setup_data_callbacks(
            stage, self.trainer.log_dir
        )
        self.trainer._callback_connector._attach_model_logging_functions()
        self.trainer.callbacks = (
            self.trainer._callback_connector._reorder_callbacks(
                self.trainer.callbacks
            )
        )


# class QDTrackCLI(BaseCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.link_arguments("data.experiment", "model.experiment")


if __name__ == "__main__":
    BaseCLI(
        model_class=QDTrack,
        datamodule_class=QDTrackDataModule,
        trainer_class=DefaultTrainer,
        save_config_overwrite=True,
    )
