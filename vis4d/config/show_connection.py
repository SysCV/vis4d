"""Show connected components in the config."""

from __future__ import annotations

import inspect
from typing import Any, TypedDict, get_type_hints

from absl import app  # pylint: disable=no-name-in-module
from torch import nn

from vis4d.common.typing import ArgsType
from vis4d.engine.callbacks import (
    Callback,
    EvaluatorCallback,
    VisualizerCallback,
)
from vis4d.engine.connectors import CallbackConnector, DataConnector
from vis4d.engine.flag import _CONFIG
from vis4d.engine.loss_module import LossModule
from vis4d.eval.base import Evaluator
from vis4d.vis.base import Visualizer

from .config_dict import instantiate_classes


# Types
class DataConnectionInfo(TypedDict):
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

    Requires "forward_train" and "forward_test" to be defined and properly
    typed!

    Args:
        model: Model to extract data from

    Returns:
        train_connections, test_connections
    """
    train_t = get_type_hints(model.forward_train)["return"]
    test_t = get_type_hints(model.forward_test)["return"]

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
    return {"train": train_connection_info, "test": test_connection_info}


def _get_loss_connection_infos(loss: LossModule) -> list[DataConnectionInfo]:
    """Returns the connection infos for a loss.

    Args:
        loss (LossModule): Custom loss module with .forward()

    Returns:
        DataConnectionInfo for the loss.
    """
    loss_connection_info = []
    for l in loss.losses:
        loss_out = []
        loss_in = []
        for entry, value in l["connector"].key_mapping.items():
            loss_out.append(f"{entry}")
            loss_in.append(f"<{_rename_ds(value['source'])}>-" + value["key"])

        loss_connection_info.append(
            DataConnectionInfo(
                in_keys=loss_in, out_keys=loss_out, name=l["name"]
            )
        )

    return loss_connection_info


def _get_vis_connection_infos(
    visualizer: Visualizer,
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


def _get_data_connector_infos(
    data_connector: DataConnector, name: str
) -> DataConnectionInfo:
    """Returns the connection infos for a DataConnector.

    Args:
        data_connector (DataConnector): Data connector to extract data.
        name (str): Name of the data connector.

    Returns:
        DataConnectionInfo for the data connector.
    """
    return DataConnectionInfo(
        in_keys=["<d>-" + e for e in list(data_connector.key_mapping.keys())],
        out_keys=list(data_connector.key_mapping.values()),
        name=name,
    )


def _get_cb_connection_infos(
    name: str,
    cb_data_connector: None | CallbackConnector = None,
) -> DataConnectionInfo | None:
    """Returns the connection infos for a callback."""
    if cb_data_connector is not None:
        eval_out = []
        eval_in = []
        for entry, value in cb_data_connector.key_mapping.items():
            eval_out.append(f"{entry}")
            eval_in.append(f"<{_rename_ds(value['source'])}>-" + value["key"])
        return DataConnectionInfo(
            in_keys=eval_in, out_keys=eval_out, name=name
        )
    return None


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


def prints_datagraph_for_config(
    model: nn.Module,
    train_data_connector: DataConnector,
    test_data_connector: DataConnector,
    loss: LossModule,
    callbacks: list[Callback],
) -> str:
    """Shows the setup of the configuration objects.

    For each components, plots which inputs is connected to which output.
    Connected components are marked with "*". Use this to debug your
    configuration setup.

    Note, that data loaded from the dataset are highlighted with <d> and data
    from model predictions with <p>.

    Args:
        model (nn.Module): Model to plot.
        train_data_connector (DataConnector): Train data connector to plot.
        test_data_connector (DataConnector): Test data connector to plot.
        loss (LossModule): Loss to plot.
        callbacks (list[Callback]): Callbacks to plot.

    Returns:
        str: The datagraph as a string, that can be printed to the console.

    Example:
        The following is train datagraph for FasterRCNN with COCO.
        Inputs loaded from dataset are marked with <d> and predictions
        with <p>. Unconnected inputs are missing a (*) sign.

        >>> dg = prints_datagraph_for_config(model, train_data_connector, test_data_connector, loss, callbacks)))
        >>> print(dg)
        ```
        # TODO: check if this is correct
        ===================================
        =          Training Loop          =
        ===================================
                            --------------
                <d>-boxes2d |            | *boxes2d
        <d>-boxes2d_classes |            | *boxes2d_classes
                 <d>-images | Train Data | *images
               <d>-input_hw |            | *input_hw
                            --------------
                         --------------
                boxes2d* |            | <p>-proposals
        boxes2d_classes* |            | <p>-roi
                 images* |            | *<p>-rpn
               input_hw* | FasterRCNN | <p>-sampled_proposals
             original_hw |            | <p>-sampled_target_indices
                         |            | <p>-sampled_targets
                         --------------
                         -----------
            <p>-rpn.cls* |         | cls_outs
            <d>-input_hw |         | images_hw
            <p>-rpn.box* | RPNLoss | reg_outs
             <d>-boxes2d |         | target_boxes
                         -----------
                                        ------------
            <p>-sampled_proposals.boxes |          | boxes
             <p>-sampled_targets.labels |          | boxes_mask
                      <p>-roi.cls_score |          | class_outs
                      <p>-roi.bbox_pred | RCNNLoss | regression_outs
              <p>-sampled_targets.boxes |          | target_boxes
            <p>-sampled_targets.classes |          | target_classes
                                        ------------
        ===================================
        =          Testing Loop           =
        ===================================
                        -------------
             <d>-images |           | *images
           <d>-input_hw | Test Data | *input_hw
        <d>-original_hw |           | *original_hw
                        -------------
                        --------------
                boxes2d |            | <p>-boxes
        boxes2d_classes |            | <p>-class_ids
                images* | FasterRCNN | <p>-scores
              input_hw* |            |
           original_hw* |            |
                        --------------
        ===================================
        =            Callbacks            =
        ===================================
                            -------------------------
        <d>-original_images |                       | *images
           <d>-sample_names |                       | *image_names
                  <p>-boxes | BoundingBoxVisualizer | *boxes
                 <p>-scores |                       | *scores
              <p>-class_ids |                       | *class_ids
                            -------------------------
                         ----------------------
        <d>-sample_names |                     | *coco_image_id
               <p>-boxes |                     | *pred_boxes
              <p>-scores | COCODetectEvaluator | *pred_scores
           <p>-class_ids |                     | *pred_classes
                         ----------------------
    ```
    """
    model_connection_info = _get_model_conn_infos(model)

    # TODO: support more data connectors
    assert isinstance(train_data_connector, DataConnector) and isinstance(
        test_data_connector, DataConnector
    ), "Only DataConnector is supported."
    train_data_connection_info = _get_data_connector_infos(
        train_data_connector, name="Train Data"
    )
    test_data_connection_info = _get_data_connector_infos(
        test_data_connector, name="Test Data"
    )

    loss_info = _get_loss_connection_infos(loss)
    log_str = ""

    # connect components
    log_str += "=" * 35 + "\n"
    log_str += "=" + " " * 10 + "Training Loop" + " " * 10 + "=" + "\n"
    log_str += "=" * 35 + "\n"

    train_components = [
        train_data_connection_info,
        model_connection_info["train"],
    ] + loss_info

    for inp, out in zip(train_components[:-1], train_components[1:]):
        connect_components(inp, out)
    for e in train_components:
        log_str += print_box(e["name"], e["in_keys"], e["out_keys"])

    log_str += "=" * 35 + "\n"
    log_str += "=" + " " * 10 + "Testing Loop " + " " * 10 + "=" + "\n"
    log_str += "=" * 35 + "\n"

    test_components = [
        test_data_connection_info,
        model_connection_info["test"],
    ]

    for inp, out in zip(test_components[:-1], test_components[1:]):
        connect_components(inp, out)

    for e in test_components:
        log_str += print_box(e["name"], e["in_keys"], e["out_keys"])

    # TODO: Add support for more callbacks and handle train_connector
    log_str += "=" * 35 + "\n"
    log_str += "=" + " " * 12 + "Callbacks" + " " * 12 + "=" + "\n"
    log_str += "=" * 35 + "\n"

    # evaluator and visualizer
    callback_components: list[DataConnectionInfo] = []

    for cb in callbacks:
        if isinstance(cb, EvaluatorCallback):
            evaluator = cb.evaluator

            connect_info = _get_evaluator_connection_infos(evaluator)
            component = _get_cb_connection_infos(
                cb.evaluator.__class__.__name__, cb.test_connector
            )

            # found matching connector
            if component is not None:
                connect_components(component, connect_info)
                callback_components.append(component)

        if isinstance(cb, VisualizerCallback):
            visualizer = cb.visualizer

            connect_info = _get_vis_connection_infos(visualizer)

            component = _get_cb_connection_infos(
                cb.visualizer.__class__.__name__, cb.test_connector
            )

            # found matching connector
            if component is not None:
                connect_components(component, connect_info)
                callback_components.append(component)

    for e in callback_components:
        log_str += print_box(e["name"], e["in_keys"], e["out_keys"])

    return log_str


def main(
    argv: ArgsType,  # pylint: disable=unused-argument
) -> None:  # pragma: no cover
    """Main entry point to show connected components in the config.

    >>> python -m vis4d.config.show_connection --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py
    """
    config = _CONFIG.value

    train_data_connector = instantiate_classes(config.train_data_connector)
    test_data_connector = instantiate_classes(config.test_data_connector)
    loss = instantiate_classes(config.loss)
    model = instantiate_classes(config.model)
    callbacks = [instantiate_classes(cb) for cb in config.callbacks]

    dg = prints_datagraph_for_config(
        model, train_data_connector, test_data_connector, loss, callbacks
    )
    print(dg)


if __name__ == "__main__":  # pragma: no cover
    app.run(main)
