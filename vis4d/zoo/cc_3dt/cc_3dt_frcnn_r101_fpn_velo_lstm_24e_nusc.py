# pylint: disable=duplicate-code
"""CC-3DT inference with Faster-RCNN ResNet-101 detector using VeloLSTM."""
from __future__ import annotations

import pytorch_lightning as pl

from vis4d.config import class_config
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.datasets.nuscenes import (
    NuScenes,
    nuscenes_class_map,
    nuscenes_detection_range_map,
)
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import (
    MultiSensorCallbackConnector,
    MultiSensorDataConnector,
)
from vis4d.eval.nuscenes import (
    NuScenesDet3DEvaluator,
    NuScenesTrack3DEvaluator,
)
from vis4d.model.motion.velo_lstm import VeloLSTM
from vis4d.op.base import ResNet
from vis4d.zoo.cc_3dt.data import (
    CONN_NUSC_DET3D_EVAL,
    CONN_NUSC_TRACK3D_EVAL,
    get_nusc_cfg,
)
from vis4d.zoo.cc_3dt.model import CONN_BBOX_3D_TEST, get_cc_3dt_cfg


def get_config() -> ExperimentConfig:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(
        exp_name="cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc"
    )

    config.velo_lstm_weights = "https://dl.cv.ethz.ch/vis4d/cc_3dt/velo_lstm_cc_3dt_frcnn_r101_fpn_100e_nusc.pt"  # pylint: disable=line-too-long

    # Hyper Parameters
    config.params = ExperimentParameters()

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/nuscenes"
    version = "v1.0-trainval"
    train_split = "train"
    test_split = "val"

    data_backend = class_config(HDF5Backend)

    config.data = get_nusc_cfg(
        data_root=data_root,
        version=version,
        train_split=train_split,
        test_split=test_split,
        data_backend=data_backend,
        # batch_normalize_images=True,  # Turn on for using old checkpoints
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet101", pretrained=True, trainable_layers=3
    )

    nuscenes_detection_range = [
        nuscenes_detection_range_map[k] for k in nuscenes_class_map
    ]

    config.model, config.loss = get_cc_3dt_cfg(
        num_classes=len(nuscenes_class_map),
        basemodel=basemodel,
        detection_range=nuscenes_detection_range,
        motion_model="VeloLSTM",
        lstm_model=class_config(VeloLSTM, weights=config.velo_lstm_weights),
        fps=2,
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = []

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.test_data_connector = class_config(
        MultiSensorDataConnector,
        key_mapping=CONN_BBOX_3D_TEST,
        sensors=NuScenes.CAMERAS,
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
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_DET3D_EVAL,
                sensors=NuScenes.CAMERAS,
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
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_TRACK3D_EVAL,
                sensors=NuScenes.CAMERAS,
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
