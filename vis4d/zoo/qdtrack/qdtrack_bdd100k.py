# pylint: disable=duplicate-code
"""QDTrack BDD100K."""
from __future__ import annotations

import lightning.pytorch as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

from vis4d.config import class_config
from vis4d.config.common.datasets.bdd100k import (
    CONN_BDD100K_TRACK_EVAL,
    get_bdd100k_track_cfg,
)
from vis4d.config.common.models.faster_rcnn import (
    get_default_rcnn_box_codec_cfg,
    get_default_rpn_box_codec_cfg,
)
from vis4d.config.common.models.qdtrack import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_ROI_LOSS_2D,
    CONN_RPN_LOSS_2D,
    CONN_TRACK_LOSS_2D,
)
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.config.util import (
    get_callable_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.data.datasets.bdd100k import bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    LossConnector,
)
from vis4d.engine.loss_module import LossModule
from vis4d.eval.bdd100k import BDD100KTrackEvaluator
from vis4d.model.track.qdtrack import FasterRCNNQDTrack
from vis4d.op.box.anchor.anchor_generator import AnchorGenerator
from vis4d.op.detect.rcnn import RCNNLoss
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.loss.common import smooth_l1_loss
from vis4d.op.track.qdtrack import QDTrackInstanceSimilarityLoss


def get_config() -> ExperimentConfig:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="qdtrack_frcnn_r50_fpn_bdd100k")

    # High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.02
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    config.data = get_bdd100k_track_cfg(
        data_backend=data_backend,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    num_classes = len(bdd100k_track_map)

    config.model = class_config(
        FasterRCNNQDTrack,
        num_classes=num_classes,
        # weights="https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.pt",  # pylint: disable=line-too-long
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    anchor_generator = class_config(
        AnchorGenerator,
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
    )

    rpn_box_encoder, _ = get_default_rpn_box_codec_cfg()
    rcnn_box_encoder, _ = get_default_rcnn_box_codec_cfg()

    rpn_loss = class_config(
        RPNLoss,
        anchor_generator=anchor_generator,
        box_encoder=rpn_box_encoder,
        loss_bbox=get_callable_cfg(smooth_l1_loss, beta=1.0 / 9.0),
    )
    rcnn_loss = class_config(
        RCNNLoss,
        box_encoder=rcnn_box_encoder,
        num_classes=num_classes,
        loss_bbox=get_callable_cfg(smooth_l1_loss),
    )

    track_loss = class_config(QDTrackInstanceSimilarityLoss)

    config.loss = class_config(
        LossModule,
        losses=[
            {
                "loss": rpn_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_RPN_LOSS_2D
                ),
            },
            {
                "loss": rcnn_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_ROI_LOSS_2D
                ),
            },
            {
                "loss": track_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_TRACK_LOSS_2D
                ),
            },
        ],
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(LinearLR, start_factor=0.1, total_iters=1000),
                    end=1000,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(MultiStepLR, milestones=[8, 11], gamma=0.1),
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TEST
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
                BDD100KTrackEvaluator,
                annotation_path="data/bdd100k/labels/box_track_20/val/",
            ),
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BDD100K_TRACK_EVAL
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    config.pl_trainer = pl_trainer

    pl_trainer.gradient_clip_val = 35

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
