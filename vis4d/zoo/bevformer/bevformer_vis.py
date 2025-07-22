"""BEVFormer Visualizaion for NuScenes Example."""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.engine.callbacks import VisualizerCallback
from vis4d.engine.connectors import MultiSensorCallbackConnector
from vis4d.vis.image.bbox3d_visualizer import MultiCameraBBox3DVisualizer
from vis4d.zoo.base import get_default_callbacks_cfg
from vis4d.zoo.bevformer.bevformer_base import (
    get_config as get_bevformer_config,
)
from vis4d.zoo.bevformer.data import (
    CONN_NUSC_BBOX_3D_VIS,
    NUSC_CAMERAS,
    nuscenes_class_map,
)


def get_config() -> ExperimentConfig:
    """Returns the config dict for BEVFormer on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_bevformer_config().ref_mode()

    config.experiment_name = "bevformer_vis"

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg()

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                MultiCameraBBox3DVisualizer,
                cat_mapping=nuscenes_class_map,
                width=2,
                camera_near_clip=0.15,
                cameras=NUSC_CAMERAS,
                vis_freq=1,
                plot_trajectory=False,
            ),
            save_prefix=config.output_dir,
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_BBOX_3D_VIS,
            ),
        )
    )

    config.callbacks = callbacks

    return config.value_mode()
