"""Example to show connected components in the config."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import copy
import inspect
import typing
from typing import Any

from ml_collections import ConfigDict
from torch import nn

from vis4d.config.default.connectors import default_detection_connector
from vis4d.config.util import class_config, instantiate_classes
from vis4d.engine.connectors import SourceKeyDescription, StaticDataConnector
from vis4d.model.detect.faster_rcnn import FasterRCNN


def _get_with_color(key: str, warn_unconnected: bool = True) -> str:
    """Prepends colors for internal vsiualization."""
    if "*" in key:
        # We connected this one
        return f"\033[94m{key}\033[00m"
    if "<d>" in key:  # key comes from data
        return f"\033[90m{key}\033[00m"

    # comes from prediction and is not connected
    if warn_unconnected:
        return f"\u001b[33m{key}\033[00m"
    else:
        return f"\033[00m{key}\033[00m"


def print_box(title: str, inputs: list[str], outputs: list[str]) -> None:
    """Prints a box with title and in/outputs.

    Args:
        title: Title to plot in the middle.
        inputs: inputs to plot on the left.
        outputs: Outputs to plot on the right.

    Example:
                            --------------
                <d>-boxes2d |            | *boxes2d
        <d>-boxes2d_classes |            | *boxes2d_classes
                 <d>-images | Train Data | *images
               <d>-input_hw |            | *input_hw
                            --------------
    """
    len_title = len(title) + 4

    n_lines = max(len(inputs), len(outputs))

    max_len_inputs = max([0] + [len(inp) for inp in inputs])
    max_len_outputs = max([0] + [len(out) for out in outputs])

    divider = (
        " " * (max_len_inputs + 1)
        + "-" * len_title
        + " " * (max_len_outputs + 1)
    )
    lines = divider + "\n"
    for idx in range(n_lines):
        in_data = inputs[idx] if len(inputs) > idx else ""
        # left pad
        in_key = " " * (max_len_inputs - len(in_data)) + in_data

        out_data = outputs[idx] if len(outputs) > idx else ""
        # left pad
        out_key = out_data + " " * (max_len_outputs - len(out_data))

        # title in middle
        line = ""
        line += _get_with_color(in_key)
        line += " | "
        line += " " * len(title) if idx != n_lines // 2 else title
        line += " | "
        line += _get_with_color(out_key)

        lines += line + "\n"

    lines += divider + "\n"
    print(lines)


def resolve_named_tuple(  # type:ignore
    clazz: Any, prefix: str = ""
) -> list[str]:
    """Returns all fields defined in the clazz t.

    Use this to get all fields defined for an e.g. Named Tuple.

    Args:
        clazz: Class that should be resolved
        prefix: Prefix to prepend (will be prefix.<field>)

    Returns:
        List with all fields and prefixes prepended.
    """
    fields = []
    if hasattr(clazz, "_fields"):
        for f in clazz._fields:
            p = f"{prefix}.{f}" if len(prefix) > 0 else f
            fields += resolve_named_tuple(getattr(clazz, f), prefix=p)
        return fields
    else:
        return [prefix]


class DataConnectionInfo(typing.TypedDict):
    """Internal type def for visualization.

    This defines a block component
    """

    in_keys: list[str]
    out_keys: list[str]
    name: str


def connect_components(
    in_info: DataConnectionInfo, out_info: DataConnectionInfo
) -> None:
    """Marks two components as connected.

    Checks if they have intersecting keys and marks them as matched.
    Updates the components inplace.

    Args:
        in_info (DataConnectionInfo): Input DataConnection
        out_info (DataConnectionInfo): Ouput DataConnection
    """
    out_keys = []
    for out in out_info["in_keys"]:
        out = out.replace("*", "")
        out_keys.append(out.split(".")[0])

    # Check connection
    for idx, key in enumerate(in_info["out_keys"]):
        key = key.replace("*", "")
        for o_idx, o_key in enumerate(out_keys):
            if key == o_key:
                in_info["out_keys"][idx] = "*" + key
                out_info["in_keys"][o_idx] = (
                    " " + out_info["in_keys"][o_idx].replace("*", "") + "*"
                )


def _rename_ds(name: str) -> str:
    """Replaces data with d and prediction with p.

    Use this to remap the datasources to shorter names.

    Args:
        name: Name to remap

    Returns:
        remapped name
    """
    return name.replace("data", "d").replace("prediction", "p")


def _get_model_conn_infos(
    model: nn.Module,
) -> dict[str, DataConnectionInfo]:
    """Returns the connection infos for a pytorch Model.

    Requires "forward_train" and "forward_test" to be defined and properly typed!

    Args:
        model: Model to extract data from

    Returns:
        train_connections, test_connections
    """
    train_t = typing.get_type_hints(model.forward_train)["return"]
    test_t = typing.get_type_hints(model.forward_test)["return"]

    train_connection_info = DataConnectionInfo(
        in_keys=sorted(
            list(inspect.signature(model.forward).parameters.keys())
        ),
        out_keys=[
            "<p>-" + e for e in sorted(resolve_named_tuple(train_t, prefix=""))
        ],
        name=model.__class__.__name__,
    )

    test_connection_info = DataConnectionInfo(
        in_keys=sorted(
            list(inspect.signature(model.forward).parameters.keys())
        ),
        out_keys=[
            "<p>-" + e for e in sorted(resolve_named_tuple(test_t, prefix=""))
        ],
        name=model.__class__.__name__,
    )
    return dict(train=train_connection_info, test=test_connection_info)


def _get_static_connector_infos(
    data_connector: StaticDataConnector,
) -> dict[str, DataConnectionInfo]:
    # train
    train_connection_info = DataConnectionInfo(
        in_keys=[
            "<d>-" + e
            for e in list(data_connector.connections["train"].keys())
        ],
        out_keys=list(data_connector.connections["train"].values()),
        name="Train Data",
    )
    # test
    test_connection_info = DataConnectionInfo(
        in_keys=[
            "<d>-" + e for e in list(data_connector.connections["test"].keys())
        ],
        out_keys=list(data_connector.connections["test"].values()),
        name="Test Data",
    )

    # loss
    loss_out = []
    loss_in = []
    for entry, value in data_connector.connections["loss"].items():
        loss_out.append(f"{entry}")
        loss_in.append(f"<{_rename_ds(value['source'])}>-" + value["key"])

    loss_connection_info = DataConnectionInfo(
        in_keys=loss_in, out_keys=loss_out, name="Loss Connector"
    )
    return dict(
        train=train_connection_info,
        test=test_connection_info,
        loss=loss_connection_info,
    )


def show_config_connections(instantiated_config: ConfigDict) -> None:
    """Shows the setup of the configuration objects.

    For each components, plots which inputs is connected to which output.
    Connected components are marked with "*". Use this to debug your
    configuration setup.

    Note, that data loaded from the dataset are highlighted with <d> and data
    from model predictions with <p>.

    Args:
        instantiated_config (ConfigDict): The instantiated Config Dict.
    """
    model = instantiated_config.model
    data_connector = instantiated_config.data_connector

    model_connection_info = _get_model_conn_infos(model)
    assert isinstance(data_connector, StaticDataConnector)
    data_connection_info = _get_static_connector_infos(data_connector)

    # connect components
    print("=" * 40)
    print("=", " " * 10, "Training Loop ", " " * 10, "=")
    print("=" * 40)
    train_components = copy.deepcopy(
        [
            data_connection_info["train"],
            model_connection_info["train"],
            data_connection_info["loss"],
        ]
    )
    for inp, out in zip(train_components[:-1], train_components[1:]):
        connect_components(inp, out)
    for e in train_components:
        print_box(e["name"], e["in_keys"], e["out_keys"])

    print("=" * 40)
    print("=", " " * 10, "Testing Loop ", " " * 10, "=")
    print("=" * 40)
    train_components = copy.deepcopy(
        [
            data_connection_info["test"],
            model_connection_info["test"],
        ]
    )
    for inp, out in zip(train_components[:-1], train_components[1:]):
        connect_components(inp, out)
    for e in train_components:
        print_box(e["name"], e["in_keys"], e["out_keys"])


############## Create Config #######################3
config = ConfigDict()
coco_eval = dict()
coco_eval["coco_image_id"] = SourceKeyDescription(
    key="coco_image_id", source="data"
)
coco_eval["pred_boxes"] = SourceKeyDescription(
    key="pred_boxes", source="prediction"
)
coco_eval["pred_scores"] = SourceKeyDescription(
    key="pred_scores", source="prediction"
)
coco_eval["pred_classes"] = SourceKeyDescription(
    key="pred_classes", source="prediction"
)
data_connector_cfg = default_detection_connector(dict(coco=coco_eval))
config.data_connector = data_connector_cfg
config.model = class_config(FasterRCNN, num_classes=80, weights=None)

########## Show connected components ##################

show_config_connections(instantiate_classes(config))
