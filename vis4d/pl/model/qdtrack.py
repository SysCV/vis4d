# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Default run configuration for pytorch lightning."""
from torch.utils.data import DataLoader

from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE
from vis4d.common.typing import MetricLogs
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.bdd100k import BDD100K, bdd100k_track_map
from vis4d.engine.data.detect import (
    default_test_pipeline,
    default_train_pipeline,
)
from vis4d.eval import Evaluator
from vis4d.model.track.qdtrack import FasterRCNNQDTrack
from vis4d.pl.data.base import DataModule
from vis4d.pl.defaults import sgd, step_schedule

from ..optimizer import DefaultOptimizer
from ..trainer import CLI

if SCALABEL_AVAILABLE and BDD100K_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.eval.mot import acc_single_video_mot, evaluate_track
    from scalabel.label.io import group_and_sort, load
    from scalabel.label.transforms import xyxy_to_box2d
    from scalabel.label.typing import Frame, Label

from vis4d.data.io.hdf5 import HDF5Backend


class ScalabelEvaluator(Evaluator):

    inverse_track_map = {v: k for k, v in bdd100k_track_map.items()}

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return f"ScalabelEvaluator"

    @property
    def metrics(self) -> list[str]:
        return ["track"]

    def reset(self) -> None:
        self.frames = []

    def process(self, data, outputs) -> None:
        for i, output in enumerate(outputs):
            labels = []
            for box, score, class_id, track_id in zip(
                output.boxes, output.scores, output.class_ids, output.track_ids
            ):
                box2d = xyxy_to_box2d(*box.cpu().numpy().tolist())
                label = Label(
                    box2d=box2d,
                    category=self.inverse_track_map[int(class_id)],
                    score=float(score),
                    id=str(int(track_id)),
                )
                labels.append(label)
            frame_id = data[CommonKeys.frame_ids][i]
            frame = Frame(
                name=data["name"][i],
                videoName=data["videoName"][i],
                frameIndex=frame_id,
                labels=labels,
            )
            self.frames.append(frame)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        annotation_path = "data/bdd100k/labels/box_track_20/val/"
        if metric == "track":
            bdd100k_anns = load(annotation_path)
            frames = bdd100k_anns.frames
            bdd100k_cfg = load_bdd100k_config("box_track")
            scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
            results = evaluate_track(
                acc_single_video_mot,
                gts=group_and_sort(scalabel_frames),
                results=group_and_sort(self.frames),
                config=bdd100k_cfg.scalabel,
                nproc=0,
            )
            return {}, results
        else:
            raise NotImplementedError


2


class TrackDataModule(DataModule):
    """Track data module."""

    def train_dataloader(self) -> DataLoader:
        """Setup training data pipeline."""
        if self.experiment == "bdd100k":
            dataloader = default_train_pipeline(
                BDD100K(
                    "data/bdd100k/images/track/train/",
                    "data/bdd100k/labels/box_track_20/train/",
                    config_path="box_track",
                ),
                self.samples_per_gpu,
                self.workers_per_gpu,
                (720, 1280),
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloader

    def test_dataloader(self) -> list[DataLoader]:
        """Setup inference pipeline."""
        if self.experiment == "bdd100k":
            dataloaders = default_test_pipeline(
                BDD100K(
                    "data/bdd100k/images/track/val/",
                    "data/bdd100k/labels/box_track_20/val/",
                    config_path="box_track",
                    data_backend=HDF5Backend(),
                ),
                self.samples_per_gpu,
                self.workers_per_gpu,
                (720, 1280),
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloaders

    def evaluators(self) -> list[Evaluator]:
        """Define evaluators associated with test datasets."""
        if self.experiment == "bdd100k":
            evaluators = [ScalabelEvaluator()]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return evaluators


def qdtrack_connect(mode: str, data):
    if mode == "test":
        return dict(
            images=data[CommonKeys.images],
            images_hw=data[CommonKeys.input_hw],
            frame_ids=data[CommonKeys.frame_ids],
        )
    raise NotImplementedError


def setup_model(  # pylint: disable=invalid-name
    experiment: str,
    lr: float = 0.02,
    max_epochs: int = 12,
) -> DefaultOptimizer:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        num_classes = len(bdd100k_track_map)
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    model = FasterRCNNQDTrack(num_classes=num_classes)

    loss = None

    return DefaultOptimizer(
        model,
        loss,
        data_connector=qdtrack_connect,
        optimizer_init=sgd(lr),
        lr_scheduler_init=step_schedule(max_epochs),
    )


class DefaultCLI(CLI):
    """Default CLI for running models with pytorch lightning."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    # pylint: disable=pointless-string-statement
    """Main function.

    Example Usage:
    >>> python -m vis4d.pl.model.qdtrack test \
        --trainer.exp_name qdtrack \
        --trainer.accelerator gpu --trainer.devices 1 \
        --data.experiment bdd100k \
        --data.samples_per_gpu 1 --data.workers_per_gpu 4"""
    DefaultCLI(model_class=setup_model, datamodule_class=TrackDataModule)
