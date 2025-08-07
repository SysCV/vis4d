# Config

We provide a simple and flexible config system that allows you to easily define experiments as well as create new models, datasets, and other components.

For this, we build up on [ml_collections](https://github.com/google/ml_collections) to provide a simple and flexible config system.

While it is possible to create configs using yaml files, we recommend using the provided python API to create configs.

Using the python API allows you to use the IDE to autocomplete config fields and allows to utilize pythons built-in import system as well as type annotations.

## FieldConfigDict and Instantiate

We use [FieldConfigDict](https://github.com/SysCV/vis4d/blob/main/vis4d/config/config_dict.py) as the base class for all configs.
This class works similar to a python dictionary, but uses references instead of values to store the config values.

A key feature of the config system is the ability to instantiate configs from `FieldConfigDict`.

By defining the config in python code, we can use the IDE to autocomplete config fields and use pythons import system.
This allows us to resolve the full class and function names without having to explicitly specify the full path.

To create an instantiable config that can be used to create a MaskRCNN instance with provided kwargs, we provide the `class_config` function to convert the `MaskRCNN` to `FieldConfigDict` as follows:

```python
from vis4d.config import class_config
from vis4d.model.detect.mask_rcnn import MaskRCNN

model_cfg = class_config(MaskRCNN, num_classes=10)
```

Note that the class_config function will automatically resolve the full class or function.
This means that we can use the class name directly without having to specify the full path.

Later on, we can **instantiate** the model from the `FieldConfigDict` as:

```python
from vis4d.config import instantiate_classes

model = instantiate_classes(config)
print(type(model))
# >>> <class 'vis4d.model.detect.mask_rcnn.MaskRCNN'>
```

Alternatively, we can also use the full path to the class or function:

```python
config = class_config("vis4d.model.detect.mask_rcnn.MaskRCNN", num_classes = 10)
model = instantiate_classes(config)
```

Or directly define the config structure ourselves:

```python
from vis4d.config import FieldConfigDict

config = FieldConfigDict()

config.class_path = "vis4d.model.detect.mask_rcnn.MaskRCNN"

config.init_args = FieldConfigDict()
config.init_args.num_classes = 10

model = instantiate_classes(config)
```

## Referencing Config Fields

Another key functionality of the config system is the ability to reference other config fields.

This allows to easily reuse configs and to create complex configs that are easy to modify.

By default, all config fields will be treated as references. This means, that
changing a field in one config will also change the field in all other configs that reference it.

```python
from vis4d.config import FieldConfigDict

c1, c2 = FieldConfigDict(), FieldConfigDict()
c1.field = "test"
c2.field = c1.field
print(c1.field.get(), c2.field.get())
# >>> test test

c1.field = "changed"
print(c1.field.get(), c2.field.get())
# >>> changed changed
```

This means, that the dot operator will always return a reference to the field.

Once you are done building the config, you should call `confgi.value_mode()` to switch to value mode, which will return the actual value instead of a reference.

```python
from vis4d.config import FieldConfigDict

c1 = FieldConfigDict()
c1.field = "test"
print(c1.field)
# >>> <ml_collections.config_dict.config_dict.FieldReference object at 0x7f17e7507d60>

# Changing config dict to value mode
c1.value_mode()
print(c1.field)
# >>> "test"

# Change back to reference mode
c1.ref_mode()
print(c1.field)
# >>> <ml_collections.config_dict.config_dict.FieldReference object at 0x7f17e7507d60>
```

## Config File

Now you know our config system.
In the following, we use [`faster_rcnn_example.py`](https://github.com/SysCV/vis4d/blob/main/docs/source/user_guide/faster_rcnn_example.py) as the example to walk through the design of the config file.

Each config file is written in python, which has `get_config` function to return the `ExperimentConfig` in `value_mode`.

```python
from vis4d.config.typing import ExperimentConfig
from vis4d.zoo.base import get_default_cfg

def get_config() -> ExperimentConfig:
    """Returns the Faster-RCNN config dict for the coco detection task.

    This is an example that shows how to set up a training experiment for the
    COCO detection task.

    Returns:
        ExperimentConfig: The configuration
    """
    config = ...

    ...

    return config.value_mode()
```

This file will be parsed by our engine and be used for trainer and other components.

A Vis4D config can be parsed into the following parts:

### General Config

This part is to defined some general config settings.

```python
    ######################################################
    ##                    General Config                ##
    ######################################################
    tmpdir = tempfile.mkdtemp() # Use tmp dir for documentation

    config = get_default_cfg(
        exp_name="faster_rcnn_r50_fpn_coco", work_dir=tmpdir
    )

    # High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 1
    params.workers_per_gpu = 0
    params.lr = 0.02
    params.num_epochs = 12
    params.num_classes = 80
    config.params = params
```

- Default Config

We provide the [`get_default_cfg`](https://github.com/SysCV/vis4d/blob/main/vis4d/zoo/base/runtime.py#L15) function to help set some basic fields for the config as follow:

- `work_dir` : Working directory. Defaults is set as "vis4d-workspace".
- `experiment_name` : Defined by the user.
- `timestamp` : Timestamp when the experiment is conducted.
- `version` : By default is set to `timestamp`. User could overwrite it according to the need.
- `output_dir` : Combined as `work_dir`/`experiment_name`/`version`. The output folder for experiments' output.
- `seed` : Random seed.
- `log_every_n_steps` : logging frequency. Used by PyTorch Lightning logger (tensorboard or wandb).
- `use_tf32` : Whether to use torch TF32. [Details](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere).
- `tf32_matmul_precision` : Internal precision of float32 matrix multiplications. [Details](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision).
- `benchmark` : Whether to use PyTorch benchmark mode.
- `compute_flops` : Whether to compute model FLOPs.
- `check_unused_parameters` : Whether to check if there is unused parameters in the model. It is usefull to debug for DDP.

- Params

We will also define `params` here, which defines parameters for the experiment.

This can be used to store arbitrary values which are often modified during training.
Allowing for easy access to these values using the CLI.

Our trainer will also dump these hyperparameters as `hparams.yaml` under the `output_dir`.

### Datasets with augmentations

This part is to define training and testing data loader with datasets.

```python
    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/coco_test"
    train_split = "train"
    test_split = "train"

    config.data = get_coco_detection_cfg(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        cache_as_binary=False,
    )
```

`config.data` should have `train_dataloader` and `test_dataloader` fields and will be parsed in the engine.

Check [Data](../dev_guide/data.md) for more details.

### MODEL & LOSS

This part is to define model and training loss.

```python
    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet50", pretrained=True, trainable_layers=3
    )

    config.model, config.loss = get_faster_rcnn_cfg(
        num_classes=params.num_classes, basemodel=basemodel, weights="mmdet"
    )
```

Please check [Model & Loss](../dev_guide/model.md) for more details.

### OPTIMIZERS

we provide [`get_optimzer_cfg` and `get_lr_scheduler_cfg`](https://github.com/SysCV/vis4d/blob/main/vis4d/zoo/base/optimizer.py) to help construct the optimizer and learning rate scheduler configs.

```python
    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(
                        LinearLR, start_factor=0.001, total_iters=500
                    ),
                    end=500,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(MultiStepLR, milestones=[8, 11], gamma=0.1),
                ),
            ],
        )
    ]
```

Please check [Optimizer](../dev_guide/optimizer.md) for more details.

### DATA CONNECTOR

In Vis4D, we use data connector to connect data pipeline and model.

```python
    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector,
        key_mapping=CONN_BBOX_2D_TRAIN,
    )

    config.test_data_connector = class_config(
        DataConnector,
        key_mapping=CONN_BBOX_2D_TEST,
    )
```

`config.train_data_connector` and `config.test_data_connector` are used to define how the data should be fed to the model.

Please check [Data connector](../dev_guide/connector.md) for more details.

## CALLBACKS

We use `callbacks` to hook the functionality, e.g. visualization, evaluation, ..., etc, to trainer.

```python
    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger
    callbacks = get_default_callbacks_cfg(refresh_rate=1)

    # Evaluator
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

    config.callbacks = callbacks
```

Please check [here](../dev_guide/callback.md) for more details.

### PL CLI

We define the config of trainer and allow for easy access to the kwargs of trainer using CLI.

```python
    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    config.pl_trainer = pl_trainer
```

Please check [here](../dev_guide/trainer.md) for more details.

After setting these fields successfully, you can return the `config.value_mode()` for your `get_config` function and use our CLI to train and test your own model.

Please check out [zoo](../dev_guide/zoo.md) for more supported config.
