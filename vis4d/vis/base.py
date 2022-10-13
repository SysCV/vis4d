"""Vis4D base visualizer."""

from vis4d.common import ModelOutput
from vis4d.data.datasets.base import DictData


# TODO, this is just a proposal for a generic visualizer class
# Maybe remove, maybe combine with DataWriter
class Visualizer:
    """Base visualizer class."""

    def reset(self) -> None:
        """Reset visualizer for new round of evaluation."""
        raise NotImplementedError()

    def process(self, inputs: DictData, outputs: ModelOutput) -> None:
        """Process data of single sample."""
        raise NotImplementedError()

    def visualize(self) -> None:
        """Visualizes the stored predictions."""
        raise NotImplementedError()

    def save_to_disk(self, path_to_out_folder) -> None:
        """Saves the visualization to disk"""
        raise NotImplementedError()
