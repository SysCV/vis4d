"""Mask RCNN COCO training example."""
from __future__ import annotations

from torch import optim
from torch.optim.lr_scheduler import StepLR

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    VisualizerCallback,
)
from vis4d.config.base.datasets.coco_detection import (
    CONN_COCO_BBOX_EVAL,
    get_coco_detection_config,
)
from vis4d.config.base.models.faster_rcnn import (
    CONN_ROI_LOSS_2D,
    CONN_RPN_LOSS_2D,
)
from vis4d.config.default.data_connectors import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_BBOX_2D_VIS,
)
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.default.runtime import set_output_dir
from vis4d.config.default.sweep import linear_grid_search
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import (
    DataConnector,
    data_key,
    pred_key,
    remap_pred_keys,
)
from vis4d.engine.loss import WeightedMultiLoss
from vis4d.eval.detect.coco import COCOEvaluator
from vis4d.model.detect.mask_rcnn import MaskRCNN
from vis4d.op.detect.faster_rcnn import (
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import (
    MaskRCNNHeadLoss,
    RCNNLoss,
    SampledMaskLoss,
    positive_mask_sampler,
)
from vis4d.op.detect.rpn import RPNLoss
from vis4d.vis.image import BoundingBoxVisualizer

CONN_MASK_HEAD_LOSS_2D = {
    "mask_preds": pred_key("masks.mask_pred"),
    "target_masks": data_key("masks"),
    "sampled_target_indices": pred_key("boxes.sampled_target_indices"),
    "sampled_targets": pred_key("boxes.sampled_targets"),
    "sampled_proposals": pred_key("boxes.sampled_proposals"),
}


def get_config() -> ConfigDict:
    """Returns the config dict for the coco detection task.

    This is a simple example that shows how to set up a training experiment
    for the COCO detection task.

    Note that the high level params are exposed in the config. This allows
    to easily change them from the command line.
    E.g.:
    >>> python -m vis4d.engine.cli fit --config configs/mask_rcnn/mask_rcnn_coco.py --config.params.lr 0.001

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ConfigDict()

    config.work_dir = "vis4d-workspace"
    config.experiment_name = "mask_rcnn_coco"
    config = set_output_dir(config)

    ## High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_epochs = 10
    params.num_classes = 80
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/coco"
    train_split = "train2017"
    test_split = "val2017"

    config.data = get_coco_detection_config(
        data_root=data_root,
        train_split=train_split,
        train_keys_to_load=(
            K.images,
            K.boxes2d,
            K.boxes2d_classes,
            K.instance_masks,
        ),
        test_split=test_split,
        test_keys_to_load=(
            K.images,
            K.boxes2d,
            K.boxes2d_classes,
            K.instance_masks,
        ),
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    # Here we define the model. We use the default Faster RCNN model
    # provided by vis4d.
    config.gen = ConfigDict()
    config.gen.anchor_generator = class_config(get_default_anchor_generator)
    config.gen.rcnn_box_encoder = class_config(get_default_rcnn_box_encoder)
    config.gen.rpn_box_encoder = class_config(get_default_rpn_box_encoder)

    config.model = class_config(
        MaskRCNN,
        num_classes=params.num_classes,
        rpn_box_encoder=config.gen.rpn_box_encoder,
        rcnn_box_encoder=config.gen.rcnn_box_encoder,
        anchor_generator=config.gen.anchor_generator,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    rpn_loss = class_config(
        RPNLoss,
        anchor_generator=config.gen.anchor_generator,
        box_encoder=config.gen.rpn_box_encoder,
    )
    rcnn_loss = class_config(RCNNLoss, box_encoder=config.gen.rcnn_box_encoder)

    mask_loss = class_config(
        SampledMaskLoss,
        mask_sampler=positive_mask_sampler,
        loss=class_config(MaskRCNNHeadLoss),
    )

    config.loss = class_config(
        WeightedMultiLoss,
        losses=[
            {"loss": rpn_loss, "weight": 1.0},
            {"loss": rcnn_loss, "weight": 1.0},
            {"loss": mask_loss, "weight": 1.0},
        ],
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################

    # Here we define which optimizer to use. We use the default optimizer
    # provided by vis4d. By default, it consists of a optimizer, a learning
    # rate scheduler and a learning rate warmup and passes all the parameters
    # to the optimizer.
    # If required, we can also define multiple, custom optimizers and pass
    # them to the config. In order to only subscribe to a subset of the
    # parameters,
    #
    # We could add a filtering function as follows:
    # def only_encoder_params(params: Iterable[torch.Tensor], fun: Callable):
    #     return fun([p for p in params if "encoder" in p.name])
    #
    # config.optimizers = [
    #    get_optimizer_config(
    #        optimizer=class_config(only_encoder_params,
    #           fun=class_config(optim.SGD, lr=params.lr"))
    #        )
    #    )
    # ]

    config.optimizers = [
        get_optimizer_config(
            optimizer=class_config(optim.SGD, lr=params.lr),
            lr_scheduler=class_config(StepLR, step_size=3, gamma=0.1),
            lr_warmup=None,
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################

    # This defines how the output of each component is connected to the next
    # component. This is a very important part of the config. It defines the
    # data flow of the pipeline.
    # We use the default connections provided for mask_rcnn. Note
    # that we build up on top of the faster_rcnn losses.
    # The faster_rcnn outputs are outputted with the key "boxes" which is why
    # we need to remap the keys of the mask_rcnn losses.
    # We do this using the remap_pred_keys function.

    config.data_connector = class_config(
        DataConnector,
        train=CONN_BBOX_2D_TRAIN,
        test=CONN_BBOX_2D_TEST,
        loss={
            **remap_pred_keys(CONN_RPN_LOSS_2D, "boxes"),
            **remap_pred_keys(CONN_ROI_LOSS_2D, "boxes"),
            **CONN_MASK_HEAD_LOSS_2D,
        },
        callbacks={
            "coco_eval_test": remap_pred_keys(CONN_COCO_BBOX_EVAL, "boxes"),
            "bbox_vis_test": remap_pred_keys(CONN_BBOX_2D_VIS, "boxes"),
        },
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################

    # Here we define the evaluator. We use the default COCO evaluator for
    # bounding box detection. Note, that we need to define the connections
    # between the evaluator and the data connector in the data connector
    # section. And use the same name here.

    eval_callbacks = {
        "coco_eval": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCOEvaluator,
                data_root=config.dataset_root,
                split=config.test_split,
            ),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    ######################################################
    ##                    VISUALIZER                    ##
    ######################################################
    # Here we define the visualizer. We use the default visualizer for
    # bounding box detection. Note, that we need to define the connections
    # between the visualizer and the data connector in the data connector
    # section. And use the same name here.

    vis_callbacks = {
        "bbox_vis": class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer),
            output_dir=config.save_prefix + "/vis",
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }
    ######################################################
    ##                GENERIC CALLBACKS                 ##
    ######################################################
    # Here we define general, all purpose callbacks. Note, that these callbacks
    # do not need to be registered with the data connector.

    config.train_callbacks = {
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=config.save_prefix,
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    # Assign the defined callbacks to the config
    config.test_callbacks = {**eval_callbacks, **vis_callbacks}
    return config.value_mode()


def get_sweep() -> ConfigDict:
    """Returns the config dict for a grid search over learning rate.

    Returns:
        ConfigDict: The configuration that can be used to run a grid search.
            It can be passed to replicate_config to create a list of configs
            that can be used to run a grid search.
    """
    return linear_grid_search("params.lr", 0.001, 0.01, 3)
