# pylint: disable=duplicate-code
"""CC-3DT VeloLSTM on nuScenes."""
from __future__ import annotations

import pytorch_lightning as pl

from vis4d.config import class_config
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.typing import (
    DataConfig,
    ExperimentConfig,
    ExperimentParameters,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_class_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import (
    data_key,
    CallbackConnector,
    MultiSensorDataConnector,
)
from vis4d.eval.nuscenes import (
    NuScenesDet3DEvaluator,
    NuScenesTrack3DEvaluator,
)
from vis4d.model.track3d.cc_3dt import BEVCC3DT
from vis4d.op.base import ResNet
from vis4d.zoo.cc_3dt.data import (
    CONN_NUSC_BBOX_3D_TEST,
    CONN_NUSC_DET3D_EVAL,
    CONN_NUSC_TRACK3D_EVAL,
    get_test_dataloader,
)

CONN_NUSC_BBOX_3D_TEST = {
    "images_list": data_key(K.images, sensors=NuScenes.CAMERAS),
    "images_hw": data_key(K.original_hw, sensors=NuScenes.CAMERAS),
    "intrinsics_list": data_key(K.intrinsics, sensors=NuScenes.CAMERAS),
    "extrinsics_list": data_key(K.extrinsics, sensors=NuScenes.CAMERAS),
    "frame_ids": K.frame_ids,
    "pred_boxes3d": "pred_boxes3d",
    "pred_boxes3d_classes": "pred_boxes3d_classes",
    "pred_boxes3d_scores": "pred_boxes3d_scores",
    "pred_boxes3d_velocities": "pred_boxes3d_velocities",
}


def get_config() -> ExperimentConfig:
    """Returns the config dict for VeloLSTM on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="cc_3dt_bevdet_nusc")

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 1
    params.workers_per_gpu = 0
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/nuscenes"
    version = "v1.0-trainval"
    test_split = "val"

    data = DataConfig()

    data.train_dataloader = None

    test_dataset = class_config(
        NuScenes,
        data_root=data_root,
        version=version,
        split=test_split,
        keys_to_load=[K.images, K.original_images, K.boxes3d],
        data_backend=class_config(HDF5Backend),
        cache_as_binary=True,
        cached_file_path="data/nuscenes/bevdet_only_track_val.pkl",
    )

    data.test_dataloader = get_test_dataloader(
        test_dataset=test_dataset,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    config.data = data

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet101", pretrained=True, trainable_layers=3
    )

    config.model = class_config(BEVCC3DT, basemodel=basemodel)

    config.loss = None

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = None

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = None

    config.test_data_connector = class_config(
        MultiSensorDataConnector, key_mapping=CONN_NUSC_BBOX_3D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir)

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                NuScenesDet3DEvaluator,
                data_root=data_root,
                version=version,
                split=test_split,
            ),
            save_predictions=True,
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_NUSC_DET3D_EVAL
            ),
        )
    )

    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(NuScenesTrack3DEvaluator),
            save_predictions=True,
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_NUSC_TRACK3D_EVAL
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
