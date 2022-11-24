"""Vis4D base evaluation."""
from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from collections.abc import Callable
from typing import Any

from vis4d.common import MetricLogs
from vis4d.data.io import DataBackend, HDF5Backend


class Evaluator:
    """Abstract evaluator class."""

    @property
    def metrics(self) -> list[str]:
        """Return list of metrics to evaluate.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return []

    def gather(
        self, gather_func: Callable[[Any], Any]
    ) -> None:  # type: ignore
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

    def process(self, *args: Any) -> None:  # type: ignore
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


class SaveDataMixin:
    """Provides utility for saving predictions to file during eval."""

    def __init__(
        self,
        save_dir: None | str = None,
        data_backend: None | DataBackend = None,
    ) -> None:
        """Init.

        Args:
            save_dir (None | str, optional): Directory to save predictions
                to. If None, a temporary directory will be created. Defaults to
                None.
            data_backend (None | DataBackend, optional): Data backend. If
                None, HDF5Backend will be used. Defaults to None.
        """
        if data_backend is None:
            self.data_backend: DataBackend = HDF5Backend()
        else:
            self.data_backend = data_backend
        if save_dir is None:
            self.save_dir = tempfile.mkdtemp()
        else:
            self.save_dir = save_dir

    def save(self, data: Any, location: str) -> None:  # type: ignore
        """Save data at given relative location.

        Args:
            data (Any): Data to save.
            location (str): Location to save to, which depends on data backend.
        """
        pdata = pickle.dumps(data, protocol=-1)
        self.data_backend.set(os.path.join(self.save_dir, location), pdata)

    def get(self, location: str) -> Any:  # type: ignore
        """Get data at given relative location.

        Args:
            location (str): Location to load from, which depends on data
                backend.

        Returns:
            Any: Loaded data.
        """
        data = self.data_backend.get(os.path.join(self.save_dir, location))
        return pickle.loads(data)

    def reset(self) -> None:
        """Delete currently cached data."""
        shutil.rmtree(self.save_dir)
