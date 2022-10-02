"""Vis4D base evaluation."""
import os
import pickle
import shutil
import tempfile
from typing import Any, Callable, List, Optional, Tuple

from vis4d.data.datasets.base import DictData
from vis4d.data.io import BaseDataBackend, HDF5Backend
from vis4d.struct_to_revise.structures import MetricLogs, ModelOutput


class Evaluator:
    """Base evaluator class."""

    @property
    def metrics(self) -> List[str]:
        """Return list of metrics to evaluate."""
        return []

    def gather(self, gather_func: Callable[[Any], Any]) -> None:
        """Gather variables in case of distributed setting (if needed)."""
        pass

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        raise NotImplementedError

    def process(self, inputs: DictData, outputs: ModelOutput) -> None:
        """Process data of single sample."""
        raise NotImplementedError

    def evaluate(self, metric: str) -> Tuple[MetricLogs, str]:
        """Evaluate all predictions according to given metric.

        Returns a dictionary of scores to log and a pretty printed string.
        """
        raise NotImplementedError


class SaveDataMixin:
    """Provides utility for saving predictions to file during eval."""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        data_backend: Optional[BaseDataBackend] = None,
    ):
        """Init."""
        self.data_backend = HDF5Backend()
        if save_dir is None:
            self.save_dir = tempfile.TemporaryDirectory()
        else:
            self.save_dir = save_dir

    def save(self, data: Any, location: str) -> None:
        """Save data at given relative location."""
        pdata = pickle.dumps(data, protocol=-1)
        self.data_backend.set(os.path.join(self.save_dir, location), pdata)

    def get(self, location: str) -> Any:
        """Get data at given relative location."""
        data = self.data_backend.get(os.path.join(self.save_dir, location))
        return pickle.loads(data)

    def reset(self) -> None:
        """Delete currently cached data."""
        shutil.rmtree(self.save_dir)
