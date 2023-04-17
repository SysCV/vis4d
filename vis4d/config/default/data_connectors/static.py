"""Static data connector config."""
from __future__ import annotations

from vis4d.config import ConfigDict, class_config

from vis4d.engine.connectors import (
    DataConnectionInfo,
    StaticDataConnector,
    SourceKeyDescription,
)


def get_static_data_connector_config(
    train_connector: dict[str, str],
    test_connector: dict[str, str],
    loss_connector: dict[str, SourceKeyDescription],
    callbacks_connector: None
    | dict[str, dict[str, SourceKeyDescription]] = None,
) -> ConfigDict:
    """Get static data connector config."""
    return class_config(
        StaticDataConnector,
        connections=DataConnectionInfo(
            train=train_connector,
            test=test_connector,
            loss=loss_connector,
            callbacks=callbacks_connector,
        ),
    )
