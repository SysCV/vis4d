"""Cmd line tool for QDTrack."""
from vis4d.model import QDTrack
from projects.qdtrack.data import QDTrackDataModule
from vis4d.engine.trainer import BaseCLI, DefaultTrainer


if __name__ == "__main__":
    BaseCLI(
        model_class=QDTrack,
        datamodule_class=QDTrackDataModule,
        trainer_class=DefaultTrainer,
    )
