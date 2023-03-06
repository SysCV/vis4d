# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Default run configuration for pytorch lightning."""
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from vis4d.common import ArgsType
from vis4d.data.const import CommonKeys
from vis4d.data.datasets import VideoMixin
from vis4d.data.datasets.nuscenes import (
    NuScenes,
    nuscenes_class_range_map,
    nuscenes_track_map,
)
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    multi_sensor_collate,
)
from vis4d.data.transforms import compose
from vis4d.data.transforms.normalize import (  # normalize_image,
    batched_normalize_image,
)
from vis4d.data.transforms.pad import pad_image
from vis4d.data.transforms.resize import resize_image, resize_intrinsics
from vis4d.eval import Evaluator
from vis4d.eval.track3d.nuscenes import NuScenesEvaluator
from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT
from vis4d.pl.data.base import DataModule
from vis4d.pl.defaults import sgd, step_schedule

from ..optimizer import DefaultOptimizer
from ..trainer import CLI


class TrackDataModule(DataModule):
    """Track data module."""

    def __init__(
        self,
        *args: ArgsType,
        **kwargs: ArgsType,
    ):
        super().__init__(*args, **kwargs)
        if self.experiment == "nuscenes":
            self.version = "v1.0-trainval"
            self.split = "val"
        elif self.experiment == "nuscenes_mini":
            self.version = "v1.0-mini"
            self.split = "mini_val"

    def train_dataloader(self) -> DataLoader:
        """Setup training data pipeline."""
        raise NotImplementedError

    def test_dataloader(self) -> list[DataLoader]:
        """Setup inference pipeline."""
        if "nuscenes" in self.experiment:
            dataloaders = default_test_pipeline(
                NuScenes(
                    "data/nuscenes/",
                    version=self.version,
                    split=self.split,
                    metadata=["use_camera"],
                ),
                self.samples_per_gpu,
                self.workers_per_gpu,
                (900, 1600),
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloaders

    def evaluators(self) -> list[Evaluator]:
        """Define evaluators associated with test datasets."""
        if "nuscenes" in self.experiment:
            evaluators = [NuScenesEvaluator(self.split)]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return evaluators


def cc_3dt_connect(mode: str, data):
    if mode == "test":
        images = []
        images_hw = []
        frame_ids = []
        intrinscs = []
        extrinsics = []
        for cam in NuScenes._CAMERAS:
            images.append(data[cam][CommonKeys.images])
            images_hw.extend(data[cam][CommonKeys.original_hw])
            intrinscs.append(data[cam][CommonKeys.intrinsics])
            extrinsics.append(data[cam][CommonKeys.extrinsics])
            frame_ids.append(data[cam][CommonKeys.frame_ids])

        images = torch.cat(images, dim=0).cuda()
        intrinsics = torch.cat(intrinscs, dim=0).cuda()
        extrinsics = torch.cat(extrinsics, dim=0).cuda()
        return dict(
            images=images,
            images_hw=images_hw,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_ids=frame_ids,
        )
    raise NotImplementedError


class VideoDataPipe(DataPipe, VideoMixin):  # TODO: refactor data pipe
    def __init__(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        super().__init__(*args, **kwargs)

    @property
    def video_to_indices(self):
        return self.datasets[0].video_to_indices


def default_test_pipeline(
    datasets: Dataset | list[Dataset],
    batch_size: int,
    num_workers: int,
    im_hw: tuple[int, int],
) -> list[DataLoader]:
    """Generate default test data pipeline."""
    preprocess_fn = compose(
        [
            resize_image(
                shape=im_hw,
                keep_ratio=True,
                sensors=NuScenes._CAMERAS,
            ),
            resize_intrinsics(sensors=NuScenes._CAMERAS),
            # normalize_image(sensors=NuScenes._CAMERAS),
        ]
    )
    batchprocess_fn = compose(
        [
            pad_image(sensors=NuScenes._CAMERAS),
            batched_normalize_image(sensors=NuScenes._CAMERAS),
        ],
    )

    return build_inference_dataloaders(
        VideoDataPipe(datasets, preprocess_fn),
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        batchprocess_fn=batchprocess_fn,
        collate_fn=multi_sensor_collate,
    )


def setup_model(  # pylint: disable=invalid-name
    experiment: str,
    lr: float = 0.01,
    max_epochs: int = 12,
    detector: str = "FRCNN",
    backbone: str = "resnet101",
    motion_model: str = "KF3D",
    pure_det: bool = False,
    with_motion: bool = True,
    weights: Optional[str] = None,
) -> DefaultOptimizer:
    """Setup model with experiment specific hyperparameters."""
    if "nuscenes" in experiment:
        num_classes = len(nuscenes_track_map)
        class_range_map = torch.Tensor(nuscenes_class_range_map)
        fps = 2
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    if detector == "FRCNN":
        if with_motion:
            model = FasterRCNNCC3DT(
                num_classes=num_classes,
                backbone=backbone,
                motion_model=motion_model,
                pure_det=pure_det,
                weights=weights,
                class_range_map=class_range_map,
            )
        else:
            # TODO: add without motion functionality
            raise NotImplementedError

    loss = None

    return DefaultOptimizer(
        model,
        loss,
        data_connector=cc_3dt_connect,
        optimizer_init=sgd(lr),
        lr_scheduler_init=step_schedule(max_epochs),
    )


class DefaultCLI(CLI):
    """Default CLI for running models with pytorch lightning."""

    def add_arguments_to_parser(self, parser) -> None:
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    # pylint: disable=pointless-string-statement
    """Main function.

    Example Usage:
    >>> python -m vis4d.pl.model.cc_3dt test \
        --trainer.exp_name cc_3dt_r101_kf3d \
        --trainer.accelerator gpu --trainer.devices 1 \
        --data.experiment nuscenes_mini \
        --data.samples_per_gpu 1 --data.workers_per_gpu 4 \
        --ckpt vis4d-workspace/checkpoints/cc_3dt_R_101_FPN_nuscenes_24.ckpt
    """
    DefaultCLI(model_class=setup_model, datamodule_class=TrackDataModule)
