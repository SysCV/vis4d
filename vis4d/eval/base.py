"""Vis4D base evaluation."""
from __future__ import annotations

from vis4d.common import ArgsType, GenericFunc, MetricLogs
from vis4d.data.io import DataBackend, FileBackend


class Evaluator:  # pragma: no cover
    """Abstract evaluator class."""

    @property
    def metrics(self) -> list[str]:
        """Return list of metrics to evaluate.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return []

    def gather(self, gather_func: GenericFunc) -> None:
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError

    def process(self, *args: ArgsType) -> None:
        """Process a batch of data.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate all predictions according to given metric.

        Args:
            metric (str): Metric to evaluate.

        Raises:
            NotImplementedError: This is an abstract class method.

        Returns:
            tuple[MetricLogs, str]: Dictionary of scores to log and a pretty
                printed string.
        """
        raise NotImplementedError

    def save(self, metric: str, output_dir: str) -> None:
        """Save predictions to file.

        Args:
            metric (str): Metric to evaluate.
            output_dir (str): Directory to save predictions to.
        """


# TODO: Find a better name/place for the writer class, which is not really an
# evaluator, but is used in the same way as an evaluator. When processing a
# batch, it should be able to save the predictions to file according to the
# dataset-specific format.


class Writer(Evaluator):
    """Abstract writer class."""

    def __init__(self, backend: DataBackend = FileBackend()) -> None:
        """Init writer.

        Args:
            backend (DataBackend, optional): Data backend. Defaults to
                FileBackend().
        """
        super().__init__()
        self.backend = backend

    def reset(self) -> None:
        """Reset writer for new round of evaluation.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError

    def process(self, *args: ArgsType) -> None:
        """Process a batch of data.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """The writer does not evaluate anything."""
        pass

    def save(self) -> None:
        """Save predictions to file.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError
