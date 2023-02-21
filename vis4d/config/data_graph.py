"""Utility functions to visualize the data graph of a configuration."""
from __future__ import annotations

import copy
import inspect
import typing
from typing import Any

from ml_collections import ConfigDict
from torch import nn

from vis4d.engine.connectors import StaticDataConnector
from vis4d.engine.loss import WeightedMultiLoss
from vis4d.eval.base import Evaluator


# Types
class DataConnectionInfo(typing.TypedDict):
    """Internal type def for visualization.

    This defines a block component
    """

    in_keys: list[str]
    out_keys: list[str]
    name: str


# Private Functions
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


def _get_loss_connection_infos(loss: nn.Module) -> DataConnectionInfo:
    """Returns the connection infos for a loss.

    Args:
        loss (nn.Module): Custom loss module with .forward()

    Returns:
        DataConnectionInfo for the loss.
    """
    in_keys: set[str] = set()

    if isinstance(loss, WeightedMultiLoss):
        for l in loss.losses:
            in_keys.update(l["in_keys"])
    else:
        in_keys.update(list(inspect.signature(loss.forward).parameters.keys()))

    return DataConnectionInfo(
        in_keys=sorted(in_keys),
        out_keys=[],
        name=loss.__class__.__name__,
    )


def _get_vis_connection_infos(
    visualizer: Evaluator,
) -> DataConnectionInfo:
    """Returns the connection infos for a visualizer.

    Args:
        visualizer: Visualizer to extract data from

    Returns:
        DataConnectionInfo for the visualizer.
    """
    return DataConnectionInfo(
        in_keys=sorted(
            list(inspect.signature(visualizer.process).parameters.keys())
        ),
        out_keys=[],
        name=visualizer.__class__.__name__,
    )


def _get_evaluator_connection_infos(
    evaluator: Evaluator,
) -> DataConnectionInfo:
    """Returns the connection infos for an evaluator.

    Args:
        evaluator: Evaluator to extract data from

    Returns:
        DataConnectionInfo for the evaluator.
    """
    return DataConnectionInfo(
        in_keys=sorted(
            list(inspect.signature(evaluator.process).parameters.keys())
        ),
        out_keys=[],
        name=evaluator.__class__.__name__,
    )


def _get_static_connector_infos(
    data_connector: StaticDataConnector,
) -> dict[str, list[DataConnectionInfo]]:
    """Returns the connection infos for a StaticDataConnector.

    Args:
        data_connector: DataConnector to extract data from

    Returns:
        Dict containing train, test, loss, visualizer and evaluator connections
    """
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

    # callbacks
    callbacks: list[DataConnectionInfo] = []
    for name, evaluator in data_connector.connections.get(
        "callbacks", {}
    ).items():
        # evaluator
        eval_out = []
        eval_in = []
        for entry, value in evaluator.items():
            eval_out.append(f"{entry}")
            eval_in.append(f"<{_rename_ds(value['source'])}>-" + value["key"])
        connection_info = DataConnectionInfo(
            in_keys=eval_in, out_keys=eval_out, name=name
        )
        callbacks.append(connection_info)

    return dict(
        train=[train_connection_info],
        test=[test_connection_info],
        loss=[loss_connection_info],
        callbacks=callbacks,
    )


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


# API Functions


def print_box(
    title: str, inputs: list[str], outputs: list[str], use_color: bool = True
) -> str:
    """Prints a box with title and in/outputs.

    Args:
        title: Title to plot in the middle.
        inputs: inputs to plot on the left.
        outputs: Outputs to plot on the right.
        use_color: Whether to use color in the output.

    Returns:
        str: The box as a string.

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
        # right pad
        out_key = out_data + " " * (max_len_outputs - len(out_data))

        # title in middle
        line = ""
        line += _get_with_color(in_key)
        line += " | "
        line += " " * len(title) if idx != n_lines // 2 else title
        line += " | "
        line += _get_with_color(out_key) if use_color else out_key

        lines += line + "\n"

    lines += divider + "\n"
    return lines


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

    Examples:
        >>> Person = namedtuple("Person", ["name", "age", "gender"])
        >>> Address = namedtuple("Address", ["street", "city", "zipcode"])

        >>> resolve_named_tuple(clazz=Person, prefix="person")
        ["person.name", "person.age", "person.gender"]

        >>> resolve_named_tuple(clazz=Address, prefix="address")
        ["address.street", "address.city", "address.zipcode"]

        >>> resolve_named_tuple(clazz=Person, prefix="")
        ["name", "age", "gender"]

        With more complex types:
        >>> User = namedtuple("User", ["name", "address"])
        >>> user = User(name=Person(name="John"),  address=Address(street="str", city="zrh", zipcode="1"))

        >>> resolve_named_tuple(clazz=user, prefix="user")
        ["user.name.name", "user.address.street", "user.address.city",
         "user.address.zipcode"]



    """
    fields = []
    if hasattr(clazz, "_fields"):
        for f in clazz._fields:
            p = f"{prefix}.{f}" if len(prefix) > 0 else f
            fields += resolve_named_tuple(getattr(clazz, f), prefix=p)
        return fields
    else:
        return [prefix]


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


def prints_datagraph_for_config(instantiated_config: ConfigDict) -> str:
    """Shows the setup of the configuration objects.

    For each components, plots which inputs is connected to which output.
    Connected components are marked with "*". Use this to debug your
    configuration setup.

    Note, that data loaded from the dataset are highlighted with <d> and data
    from model predictions with <p>.

    Args:
        instantiated_config (ConfigDict): The instantiated Config Dict.

    Returns:
        str: The datagraph as a string, that can be printed to the console.

    Example:
        The following is train datagraph for FasterRCNN with COCO.
        Inputs loaded from dataset are marked with <d> and predictions
        with <p>. Unconnected inputs are missing a (*) sign.


        >>> from vis4d.config.example.faster_rcnn_coco import get_config
        >>> from vis4d.config.util import instantiate_classes
        >>> from vis4d.config.data_graph import prints_datagraph_for_config

        >>> dg = prints_datagraph_for_config(instantiate_classes(get_config()))
        >>> print(dg)
        ```
                ========================================
                =          Training Loop               =
                ========================================

                            --------------
                <d>-boxes2d |            | *boxes2d
        <d>-boxes2d_classes |            | *boxes2d_classes
                 <d>-images | Train Data | *images
               <d>-input_hw |            | *input_hw

                            --------------
                         --------------
                boxes2d* |            | <p>-proposals
        boxes2d_classes* |            | *<p>-roi
                 images* |            | *<p>-rpn
               input_hw* | FasterRCNN | *<p>-sampled_proposals
             original_hw |            | <p>-sampled_target_indices
                         |            | *<p>-sampled_targets
                         --------------

                                     ------------------
        <p>-sampled_proposals.boxes* |                | *boxes
         <p>-sampled_targets.labels* |                | *boxes_mask
                  <p>-roi.cls_score* |                | *class_outs
                        <p>-rpn.cls* |                | *cls_outs
                        <d>-input_hw |                | *images_hw
              <p>-sampled_proposals* | Loss Connector | pred_sampled_proposals
                        <p>-rpn.box* |                | *reg_outs
                  <p>-roi.bbox_pred* |                | *regression_outs
          <p>-sampled_targets.boxes* |                | *target_boxes
        <p>-sampled_targets.classes* |                | *target_classes
                                     ------------------

                         ---------------------
                 boxes* |                   |
            boxes_mask* |                   |
            class_outs* |                   |
              cls_outs* |                   |
             images_hw* |                   |
              reg_outs* | WeightedMultiLoss |
       regression_outs* |                   |
          target_boxes* |                   |
       target_class_ids |                   |
        target_classes* |                   |
                        ---------------------
    ```

    """
    model = instantiated_config.model
    data_connector = instantiated_config.data_connector

    model_connection_info = _get_model_conn_infos(model)
    assert isinstance(data_connector, StaticDataConnector)
    data_connection_info = _get_static_connector_infos(data_connector)

    loss_info = _get_loss_connection_infos(instantiated_config.loss)
    # TOOD, needs more safety checks. I.e. does config.loss exists, ...
    log_str = ""

    # connect components
    log_str += "=" * 40 + "\n"
    log_str += "=" + " " * 10 + "Training Loop " + " " * 10 + "=" + "\n"
    log_str += "=" * 40 + "\n"
    train_components = copy.deepcopy(
        [
            *data_connection_info["train"],
            model_connection_info["train"],
            *data_connection_info["loss"],
            loss_info,
        ]
    )

    for inp, out in zip(train_components[:-1], train_components[1:]):
        connect_components(inp, out)
    for e in train_components:
        log_str += print_box(e["name"], e["in_keys"], e["out_keys"])

    log_str += "=" * 40 + "\n"
    log_str += "=" + " " * 10 + "Testing Loop " + " " * 10 + "=" + "\n"
    log_str += "=" * 40 + "\n"

    train_components = copy.deepcopy(
        [
            *data_connection_info["test"],
            model_connection_info["test"],
        ]
    )
    for inp, out in zip(train_components[:-1], train_components[1:]):
        connect_components(inp, out)

    optional_components: list[DataConnectionInfo] = []
    # Connect evaluators to visualizers

    print(instantiated_config.keys())
    if "evaluators" in instantiated_config:
        for name, evaluator in instantiated_config.evaluators.items():
            print("found eavluater:", evaluator)
            connect_info = _get_evaluator_connection_infos(evaluator)

            for component in data_connection_info["evaluators"]:
                if component["name"] == name:
                    # found matching connector
                    connect_components(component, connect_info)
                    if component not in optional_components:
                        optional_components.append(component)
            optional_components.append(connect_info)

    if "visualizers" in instantiated_config:
        for name, vis in instantiated_config.visualizers.items():
            connect_info = _get_vis_connection_infos(vis)
            for component in data_connection_info["visualizers"]:
                if component["name"] == name:
                    # found matching connector
                    connect_components(component, connect_info)
                    if component not in optional_components:
                        optional_components.append(component)
            optional_components.append(connect_info)

    train_components.extend(optional_components)

    for e in train_components:
        log_str += print_box(e["name"], e["in_keys"], e["out_keys"])

    return log_str
