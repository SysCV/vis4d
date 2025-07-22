# pylint: disable=duplicate-code
"""CC-3DT with BEV detector on nuScenes."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes_detection import NuScenesDetection
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import CallbackConnector
from vis4d.eval.nuscenes import (
    NuScenesDet3DEvaluator,
    NuScenesTrack3DEvaluator,
)
from vis4d.zoo.base import get_default_callbacks_cfg
from vis4d.zoo.cc_3dt.cc_3dt_bevformer_base_velo_lstm_nusc import (
    get_config as get_cc_3dt_config,
)
from vis4d.zoo.cc_3dt.data import (
    CONN_NUSC_DET3D_EVAL,
    CONN_NUSC_TRACK3D_EVAL,
    get_test_dataloader,
)


def get_config() -> ExperimentConfig:
    """Returns the config dict for CC-3DT on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_cc_3dt_config().ref_mode()

    config.experiment_name = "cc_3dt_nusc_test"

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    config.pure_detection = ""

    data = DataConfig()

    data.train_dataloader = None

    test_dataset = class_config(
        NuScenesDetection,
        data_root="data/nuscenes",
        version="v1.0-test",
        split="test",
        keys_to_load=[K.images, K.original_images],
        data_backend=class_config(HDF5Backend),
        pure_detection=config.pure_detection,
        cache_as_binary=True,
        cached_file_path="data/nuscenes/test.pkl",
    )

    data.test_dataloader = get_test_dataloader(
        test_dataset=test_dataset, samples_per_gpu=1, workers_per_gpu=4
    )

    config.data = data

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg()

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                NuScenesDet3DEvaluator,
                data_root="data/nuscenes",
                version="v1.0-test",
                split="test",
                save_only=True,
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

    return config.value_mode()
