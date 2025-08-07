# Callbacks

We use PyTorch Lightning [Callbacks](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html#lightning.pytorch.callbacks.Callback) to support various functionality.

Each callback relies on `Callback Connector` to connect between data, model, and callbacks.

For each config, the default callback is logging callback which helps to logging the output of console.

```python
def get_default_callbacks_cfg(
    epoch_based: bool = True,
    refresh_rate: int = 50,
) -> list[ConfigDict]:
    """Get default callbacks config.

    It will return a list of callbacks config including:
        - LoggingCallback

    Args:
        epoch_based (bool, optional): Whether to use epoch based logging.
        refresh_rate (int, optional): Refresh rate for the logging. Defaults to
            50.

    Returns:
        list[ConfigDict]: List of callbacks config.
    """
    callbacks = []

    # Logger
    callbacks.append(
        class_config(
            LoggingCallback, epoch_based=epoch_based, refresh_rate=refresh_rate
        )
    )

    return callbacks
```

You can hook any `Callback` in the config as follow:

```python
callbacks.append(
    class_config(
        EvaluatorCallback,
        evaluator=class_config(
            COCODetectEvaluator, data_root=data_root, split=test_split
        ),
        metrics_to_eval=["Det"],
        test_connector=class_config(
            CallbackConnector, key_mapping=CONN_COCO_BBOX_EVAL
        ),
    )
)
```

Check more details [here](https://github.com/SysCV/vis4d/tree/main/vis4d/engine/callbacks).
