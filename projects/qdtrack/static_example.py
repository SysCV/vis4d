"""Example using static configuration files with QDTrack.

This file illustrates how static configs can be utilized to set the parameters
for a model in Vis4D.
To use this file in combination with a yaml config, run e.g.:
python static_example.py fit --config cfg.yaml
The file cfg.yaml contains an example configuration for bdd100k training.
"""
from projects.qdtrack.data import QDTrackDataModule
from vis4d.engine.trainer import BaseCLI
from vis4d.model import QDTrack

if __name__ == "__main__":
    BaseCLI(
        model_class=QDTrack,
        datamodule_class=QDTrackDataModule,
    )
