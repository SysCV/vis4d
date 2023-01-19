# type: ignore # FIXME (zrene). Remove and fix this with new engine / config
"""Pointnet and Pointnet++ training example."""
import argparse
import warnings
from typing import List, Optional, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader

from vis4d.data.const import CommonKeys
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.data.typing import DictData
from vis4d.engine.data.segment3d import (
    default_test_pipeline,
    default_train_pipeline,
)
from vis4d.engine.test import testing_loop
from vis4d.engine.train import training_loop
from vis4d.eval import Evaluator
from vis4d.eval.segment.segmentation_evaluator import SegmentationEvaluator
from vis4d.model.segment3d.pointnet import (
    PointnetSegmentationLoss,
    PointnetSegmentationModel,
)
from vis4d.model.segment3d.pointnetpp import (
    Pointnet2SegmentationLoss,
    PointNet2SegmentationModel,
)

warnings.filterwarnings("ignore")

MODEL_NAME_TO_CLS_AND_LOSS = {
    "pointnet": (PointnetSegmentationModel, PointnetSegmentationLoss),
    "pointnetpp": (PointNet2SegmentationModel, Pointnet2SegmentationLoss),
}


def get_dataloaders(
    is_training: bool = False, batch_size: int = 1, load_colors=False
) -> Tuple[Optional[DataLoader], List[DataLoader], List[Evaluator], str]:
    """Return dataloaders and evaluators."""
    data_in_keys = [CommonKeys.points3d]
    if load_colors:
        data_in_keys += [CommonKeys.colors3d]

    if is_training:
        train_loader = default_train_pipeline(
            S3DIS(
                data_root="/data/Stanford3dDataset_v1.2_Aligned_Version",
                keys_to_load=data_in_keys + [CommonKeys.semantics3d],
            ),
            batch_size,
            load_colors=load_colors,
            num_pts=4096,
        )
    else:
        train_loader = None

    test_loader = default_test_pipeline(
        S3DIS(
            data_root="/data/Stanford3dDataset_v1.2_Aligned_Version",
            split="testArea5",
            keys_to_load=data_in_keys + [CommonKeys.semantics3d],
        ),
        1,
        load_colors=load_colors,
        num_pts=4096,
    )
    test_evals = [
        SegmentationEvaluator(
            num_classes=13,
            class_mapping=dict(
                (v, k) for k, v in S3DIS.CLASS_NAME_TO_IDX.items()
            ),
        )
    ]
    test_metric = SegmentationEvaluator.METRIC_ALL
    return train_loader, test_loader, test_evals, test_metric


def data_connector(mode: str, data: DictData):
    """Data connector."""
    if mode == "train":
        data_keys = {
            CommonKeys.points3d: CommonKeys.points3d,
            CommonKeys.semantics3d: CommonKeys.semantics3d,
        }
    elif mode == "loss":
        data_keys = {CommonKeys.semantics3d: CommonKeys.semantics3d}
    else:
        data_keys = {
            CommonKeys.points3d: CommonKeys.points3d,
            # COMMON_KEYS.semantics3d: COMMON_KEYS.semantics3d,
        }
    return {v: data[k] for k, v in data_keys.items()}


def eval_connector(
    mode: str, in_data: DictData, output_data: DictData
):  # TODO, ugly
    """Data connector for evaluator and visualizer."""
    if mode == "eval":
        return {
            "prediction": output_data[CommonKeys.semantics3d],
            "groundtruth": in_data[CommonKeys.semantics3d],
        }
    return {}


def train(args: argparse.Namespace) -> None:
    """Training."""
    # parameters
    log_step = 10
    num_epochs = 400
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device("cuda")
    balance_weights = True
    save_prefix = "vis4d-workspace/test/pointnet_s3dis_epoch"

    # data loaders and evaluators
    train_loader, test_loader, test_evals, test_metric = get_dataloaders(
        True, batch_size, args.load_color
    )
    assert train_loader is not None

    # model
    in_dimension = 6  # xyz, xyz_normalized
    if args.load_color:
        in_dimension += 3  # rgb

    model_cls, loss_cls = MODEL_NAME_TO_CLS_AND_LOSS[args.model]
    segmenter = model_cls(
        num_classes=13, in_dimensions=in_dimension, weights=args.ckpt
    )
    segmenter = segmenter.to(device)

    if balance_weights:
        balanced_weigths = S3DIS.CLASS_COUNTS / S3DIS.CLASS_COUNTS.sum()
        balanced_weigths = 1 / (balanced_weigths + 0.02)
        # normalize
        balanced_weigths = (
            balanced_weigths * len(balanced_weigths) / balanced_weigths.sum()
        )
        pointnet_loss = loss_cls(semantic_weights=balanced_weigths.cuda())
    else:
        pointnet_loss = loss_cls()

    # optimization
    optimizer = optim.Adam(
        segmenter.parameters(),
        lr=learning_rate,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40], gamma=0.1
    )
    warmup = None

    visualizers = []  # [PointPredVisualizer()] if args.visualize else [] TODO

    # run training
    training_loop(
        train_loader,
        test_loader,
        test_evals,
        test_metric,
        segmenter,
        pointnet_loss,
        data_connector,
        optimizer,
        scheduler,
        num_epochs,
        log_step,
        learning_rate,
        save_prefix,
        warmup,
        visualizers=visualizers,
        eval_connector=eval_connector,
        test_every_nth_epoch=10,
        save_every_nth_epoch=5,
        vis_every_nth_epoch=-1,  # do not visualize
    )


def test(args: argparse.Namespace) -> None:
    """Testing."""
    # parameters
    device = torch.device("cuda")

    # data loaders and evaluators
    _, test_loader, test_evals, test_metric = get_dataloaders(
        load_colors=args.load_color
    )

    # model
    in_dimension = 3  # xyz
    if args.load_color:
        in_dimension += 3  # rgb

    model_cls, _ = MODEL_NAME_TO_CLS_AND_LOSS[args.model]

    segmenter = model_cls(
        num_classes=13, in_dimensions=in_dimension, weights=args.ckpt
    )
    segmenter.to(device)
    model_test_keys = [CommonKeys.points3d]

    # run testing
    testing_loop(
        test_loader,
        test_evals,
        test_metric,
        segmenter,
        data_connector,
        eval_connector,
    )
    test_evals = [e.to(device) for e in test_evals]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO train/eval.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval"
    )
    parser.add_argument("-n", "--num_gpus", default=1, help="number of gpus")
    parser.add_argument(
        "--model", default="pointnetpp", help="Name of the model "
    )
    parser.add_argument("--load_color", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--data_root",
        default="/data/Stanford3dDataset_v1.2",
        help="Path to dataset",
    )
    args = parser.parse_args()
    if args.ckpt is None:
        train(args)
    else:
        test(args)
