"""Data connector for the reference data."""
from __future__ import annotations

from vis4d.data.typing import DictData

from .base import DataConnector


class RefDataConnector(DataConnector):
    """Data connector for the reference data dict."""

    def __call__(self, data: DictData) -> DictData:
        """Returns the train input for the model."""
        return {k: [d[v] for d in data] for k, v in self.key_mapping.items()}
