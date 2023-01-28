"""Vis4D base visualizer."""

from vis4d.common.typing import ArgsType


class Visualizer:
    """Base visualizer class."""

    def reset(self) -> None:
        """Reset visualizer for new round of evaluation."""
        raise NotImplementedError()

    def process(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        """Process data of single sample."""
        raise NotImplementedError()

    def show(self, blocking: bool = True) -> None:
        """Shows the visualization.

        Args:
            blocking (bool): If the visualization should be blocking
                             and wait for human input
        """
        raise NotImplementedError()

    def save_to_disk(self, path_to_out_folder: str) -> None:
        """Saves the visualization to disk.

        Args:
            path_to_out_folder (str): Folder where the output should be written
        """
        raise NotImplementedError()
