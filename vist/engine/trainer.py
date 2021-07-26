"""DefaultTrainer for VisT."""

import pytorch_lightning as pl
from torch.utils import data

from vist.config import Config
from vist.data import build_train_dataset, build_test_dataset, TrackingInferenceSampler
from vist.model import build_model
from vist.struct import EvalResults
from vist.vis import ScalabelVisualizer

from .evaluator import ScalabelEvaluator, inference_on_dataset


def build_train_loader(cfg: Config):
    train_dataset = build_train_dataset(cfg.train)

    train_dataloader = data.DataLoader(
        train_dataset,
        cfg.solver.imgs_per_gpu,
        num_workers=cfg.solver.workers_per_gpu,
    )


def build_test_loaders(cfg: Config):
    test_dataloaders = []
    for data_cfg in cfg.test:
        dataset = build_test_dataset(data_cfg)

        sampler = (
            TrackingInferenceSampler(dataset)
            if data_cfg.inference_sampling == "sequence_based"
            else None
        )
        batch_sampler = data.sampler.BatchSampler(
            sampler, 1, drop_last=False
        )

        test_dataloader = data.DataLoader(
            dataset,
            num_workers=cfg.solver.workers_per_gpu,
            batch_sampler=batch_sampler,
            sampler=sampler,
        )
        test_dataloaders.append(test_dataloader)


def train(cfg: Config) -> None:
    """Training function."""
    # TODO setup to lightning
    # TODO dataloader needs to know expected image_channel_mode + model classes

    model = build_model(cfg.model)
    train_dataloader = build_train_loader(cfg)
    val_dataloaders = build_test_loaders(cfg)

    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, val_dataloaders)


def test(cfg: Config) -> None:
    """Test function."""

    # TODO setup to lightning

    model = build_model(cfg.model)
    # TODO load weights

    test_dataloaders = build_test_loaders(cfg)

    trainer = pl.Trainer()
    trainer.test(model, test_dataloaders)


def predict(cfg: Config) -> None:
    """Prediction function."""
    # TODO setup to lightning

    model = build_model(cfg.model)

    trainer = pl.Trainer()
    trainer.predict(model)
