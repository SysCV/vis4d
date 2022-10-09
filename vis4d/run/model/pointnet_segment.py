"""Faster RCNN COCO training example."""
import argparse
import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from vis4d.common.typing import COMMON_KEYS
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.eval import Evaluator
from vis4d.eval.s3dis import S3DisEvaluator
from vis4d.model.segment3d.pointnet import (
    PointnetSegmentationLoss,
    PointnetSegmentationModel,
)
from vis4d.optim.warmup import LinearLRWarmup
from vis4d.run.data.segment3d import (
    default_test_pipeline,
    default_train_pipeline,
)
from vis4d.run.test import testing_loop
from vis4d.run.train import training_loop

warnings.filterwarnings("ignore")


def get_dataloaders(
    is_training: bool = False, batch_size: int = 1, load_colors=False
) -> Tuple[Optional[DataLoader], List[DataLoader], List[Evaluator], str]:
    """Return dataloaders and evaluators."""

    data_in_keys = [COMMON_KEYS.points3d]
    if load_colors:
        data_in_keys += [COMMON_KEYS.colors3d]

    if is_training:
        train_loader = default_train_pipeline(
            S3DIS(
                data_root="/data/Stanford3dDataset_v1.2",
                keys_to_load=data_in_keys + [COMMON_KEYS.semantics3d],
            ),
            batch_size,
            load_colors=load_colors,
        )
    else:
        train_loader = None

    test_loader = default_test_pipeline(
        S3DIS(
            data_root="/data/Stanford3dDataset_v1.2",
            split="testArea5",
            keys_to_load=data_in_keys + [COMMON_KEYS.semantics3d],
        ),
        1,
        load_colors=load_colors,
    )
    test_evals = [S3DisEvaluator()]
    test_metric = "mIoU"
    return train_loader, test_loader, test_evals, test_metric


def train(args: argparse.Namespace) -> None:
    """Training."""
    # parameters
    log_step = 1
    num_epochs = 40
    batch_size = 16
    learning_rate = 1e-4
    device = torch.device("cuda")
    save_prefix = "vis4d-workspace/test/pointnet_s3dis_epoch"

    # data loaders and evaluators
    train_loader, test_loader, test_evals, test_metric = get_dataloaders(
        True, batch_size, args.load_color
    )
    assert train_loader is not None

    # model
    in_dimension = 3  # xyz
    if args.load_color:
        in_dimension += 3  # rgb

    segmenter = PointnetSegmentationModel(
        num_classes=13, in_dimensions=in_dimension, weights=args.ckpt
    )
    segmenter.to(device)

    pointnet_loss = PointnetSegmentationLoss()
    model_train_keys = [COMMON_KEYS.points3d, COMMON_KEYS.semantics3d]
    model_test_keys = [COMMON_KEYS.points3d]
    loss_keys = [COMMON_KEYS.semantics3d]

    # optimization
    optimizer = optim.SGD(
        segmenter.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0001,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 11], gamma=0.1
    )
    warmup = LinearLRWarmup(0.001, 500)
    # run training
    training_loop(
        train_loader,
        test_loader,
        test_evals,
        test_metric,
        segmenter,
        pointnet_loss,
        model_train_keys,
        model_test_keys,
        loss_keys,
        optimizer,
        scheduler,
        num_epochs,
        log_step,
        learning_rate,
        save_prefix,
        warmup,
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

    segmenter = PointnetSegmentationModel(
        num_classes=13, in_dimensions=in_dimension, weights=args.ckpt
    )
    segmenter.to(device)
    model_test_keys = [COMMON_KEYS.points3d]

    # run testing
    testing_loop(
        test_loader, test_evals, test_metric, segmenter, model_test_keys
    )
    test_evals = [e.to(device) for e in test_evals]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO train/eval.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval"
    )
    parser.add_argument("-n", "--num_gpus", default=1, help="number of gpus")
    parser.add_argument("--load_color", action="store_true")
    args = parser.parse_args()
    if args.ckpt is None:
        train(args)
    else:
        test(args)
