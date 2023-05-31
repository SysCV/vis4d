"""Vis4D base evaluation."""
from __future__ import annotations

from vis4d.common import ArgsType, GenericFunc, MetricLogs


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
