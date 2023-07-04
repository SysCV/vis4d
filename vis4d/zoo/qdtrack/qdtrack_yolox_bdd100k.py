# pylint: disable=duplicate-code
"""QDTrack BDD100K inference example."""
from __future__ import annotations

import pytorch_lightning as pl

from vis4d.config import class_config
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.default.data_connectors import CONN_BBOX_2D_TRACK_VIS
from vis4d.config.typing import (
    DataConfig,
    ExperimentConfig,
    ExperimentParameters,
)
from vis4d.config.util import get_inference_dataloaders_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.bdd100k import BDD100K, bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import GenerateResizeParameters, ResizeImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    data_key,
    pred_key,
)
from vis4d.eval.bdd100k import BDD100KTrackEvaluator
from vis4d.model.track.qdtrack import YOLOXQDTrack
from vis4d.vis.image import BoundingBoxVisualizer

CONN_BBOX_2D_TEST = {
    K.images: K.images,
    "images_hw": K.input_hw,
    K.original_hw: K.original_hw,
    K.frame_ids: K.frame_ids,
}

CONN_BDD100K_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "sample_names": data_key("sample_names"),
    "sequence_names": data_key("sequence_names"),
    "pred_boxes": pred_key("boxes"),
    "pred_classes": pred_key("class_ids"),
    "pred_scores": pred_key("scores"),
    "pred_track_ids": pred_key("track_ids"),
}


def get_config() -> ExperimentConfig:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="qdtrack_yolox_bdd100k")

    ckpt_path = (
        "vis4d-workspace/QDTrack/pretrained/qdtrack-yolox-ema_bdd100k.ckpt"
    )

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 4
    params.workers_per_gpu = 4
    params.lr = 0.01
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data = DataConfig()
    dataset_root = "data/bdd100k/images/track/val/"
    annotation_path = "data/bdd100k/labels/box_track_20/val/"
    config_path = "box_track"
    data_backend = class_config(HDF5Backend)

    # TODO: Add train dataset
    data.train_dataloader = None

    # Test
    test_dataset = class_config(
        BDD100K,
        data_root=dataset_root,
        keys_to_load=(K.images),
        annotation_path=annotation_path,
        config_path=config_path,
        image_channel_mode="BGR",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/track_val.pkl",
    )

    preprocess_transforms = [
        class_config(
            GenerateResizeParameters,
            shape=(800, 1440),
            keep_ratio=False,
            align_long_edge=True,
        ),
        class_config(ResizeImages),
    ]

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[class_config(PadImages), class_config(ToTensor)],
    )

    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
    )

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        samples_per_gpu=1,
        workers_per_gpu=params.workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
    )

    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    num_classes = len(bdd100k_track_map)

    config.model = class_config(
        YOLOXQDTrack, num_classes=num_classes, weights=ckpt_path
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    config.loss = None  # TODO: implement loss

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = []  # TODO: implement optimizer

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    # TODO: Add train data connector
    config.train_data_connector = None

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir)

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer, vis_freq=500),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_2D_TRACK_VIS
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KTrackEvaluator, annotation_path=annotation_path
            ),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BDD100K_EVAL
            ),
        ),
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.precision = "16-mixed"
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
