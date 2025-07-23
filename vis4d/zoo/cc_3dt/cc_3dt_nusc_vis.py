"""CC-3DT Visualizaion for NuScenes Example."""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_class_map
from vis4d.engine.callbacks import VisualizerCallback
from vis4d.engine.connectors import MultiSensorCallbackConnector
from vis4d.vis.image.bbox3d_visualizer import MultiCameraBBox3DVisualizer
from vis4d.vis.image.bev_visualizer import BEVBBox3DVisualizer
from vis4d.zoo.base import get_default_callbacks_cfg
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc import (
    get_config as get_cc_3dt_config,
)
from vis4d.zoo.cc_3dt.data import (
    CONN_NUSC_BBOX_3D_VIS,
    CONN_NUSC_BEV_BBOX_3D_VIS,
)


def get_config() -> ExperimentConfig:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_cc_3dt_config().ref_mode()

    config.experiment_name = "cc_3dt_nusc_vis"

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
                cameras=NuScenes.CAMERAS,
                vis_freq=1,
            ),
            output_dir=config.output_dir,
            save_prefix="boxes3d",
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_BBOX_3D_VIS,
            ),
        )
    )

    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BEVBBox3DVisualizer, width=2, vis_freq=1),
            output_dir=config.output_dir,
            save_prefix="bev",
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_BEV_BBOX_3D_VIS,
            ),
        )
    )

    config.callbacks = callbacks

    return config.value_mode()
