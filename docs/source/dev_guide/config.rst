######
Config
######

We provide a simple and flexible config system that allows you to easily define experiments as well as create new models, datasets, and other components.
For this, we build up on `ml_collections <https://github.com/google/ml_collections>`_ to provide a simple and flexible config system.
While it is possible to create configs using yaml files, we recommend using the provided python API to create configs.
Using the python API allows you to use the IDE to autocomplete config fields and allows to utilize pythons built-in import system as well as type annotations.

We use `FieldConfigDict <TODO>`_ as the base class for all configs. This class works similar to a python dictionary, but uses references instead of values to store the config values.

=================
Experiment Config
=================
Each experiment is defined by a config that inherits from `ExperimentConfig <TODO>`_.
A valid experiment config must define the following fields:

data
    FieldConfigDict containining test (and training) data loader.
output_dir
    Output Directory for the experiment. Log files, checkpoints and reproducible configs will be stored here.
train_data_connector
    DataConnector for the training data. This defines how the training data should be fed to the model.
test_data_connector
    DataConnector for the test data. This defines how the test data should be fed to the model.
model
    ModelConfig defining the model. It is assumed to have a `forward_train` and `forward_test` method.
optimizers
    List of optimizers to use for training.
loss
    Loss config defining the loss function to use.
callbacks
    List of callbacks to use during training.
params
    Parameters for the experiment. This can be used to store arbitrary values which are often
    modified during training. Allowing for easy access to these values using the CLI.

===================
Instantiate Configs
===================
A key feature of the config system is the ability to instantiate configs from FieldConfigDict.
By defining the config in python code, we can use the IDE to autocomplete config fields and use pythons import system.
This allows us to resolve the full class and function names without having to explicitly specify the full path.
For example, we can define a model config as follows:

.. code-block:: python

    from vis4d.config import FieldConfigDict, instantiate_classes, class_config
    from vis4d.model.detect.mask_rcnn import MaskRCNN

    # Create an instantiable config that can be used to create a MaskRCNN instance
    # With provvided kwargs
    config = class_config(MaskRCNN, num_classes = 10)
    model = instantiate_classes(config)
    print(type(model))

.. code-block:: bash

    >> <class 'vis4d.model.detect.mask_rcnn.MaskRCNN'>

Note that the class_config function will automatically resolve the full class or function.
This means that we can use the class name directly without having to specify the full path.
Alternatively, we can also use the full path to the class or function:

.. code-block:: python

    config = class_config("vis4d.model.detect.mask_rcnn.MaskRCNN", num_classes = 10)
    model = instantiate_classes(config)

Or directly define the config structure ourselves:

.. code-block:: python

    config = FieldConfigDict()
    config.class_path = "vis4d.model.detect.mask_rcnn.MaskRCNN"
    config.init_args = FieldConfigDict()
    config.init_args.num_classes = 10
    model = instantiate_classes(config)

=========================
Referencing Config Fields
=========================
A key functionality of the config system is the ability to reference other config fields.
This allows to easily reuse configs and to create complex configs that are easy to modify.

By default, all config fields will be treated as references. This means, that
changing a field in one config will also change the field in all other configs that reference it.

.. code-block:: python

    from vis4d.config import FieldConfigDict
    c1, c2 = FieldConfigDict(), FieldConfigDict()
    c1.field = "test"
    c2.field = c1.field
    print(c1.field.get(), c2.field.get())
    # >> test test
    c1.field = "changed"
    print(c1.field.get(), c2.field.get())
    # >> changed changed

This means, that the dot operator will always return a reference to the field.
Once you are done building the config, you should call `confgi.value_mode()` to switch to value mode, which will return the actual value instead of a reference.

.. code-block:: python

    from vis4d.config import FieldConfigDict
    c1 = FieldConfigDict()
    c1.field = "test"
    print(c1.field)

.. code-block:: bash

    >>  <ml_collections.config_dict.config_dict.FieldReference object at 0x7f17e7507d60>

.. code-block:: python

    # Changing config dict to value mode
    c1.value_mode()
    print(c1.field)

.. code-block:: bash

    >> "test"

.. code-block:: python

    # Change back to reference mode
    c1.ref_mode()
    print(c1.field)

.. code-block:: bash

    >>  <ml_collections.config_dict.config_dict.FieldReference object at 0x7f17e7507d60>

===============================
Callbacks and Trainer Arguments
===============================
We support custom Callbacks as well as Pytorch Lightning Trainer Arguments.

--------------------
Using the Python API
--------------------
While we provide a CLI for training and evaluating your models, you can also use the python API directly.

-----------------------
Using the Trainer class
-----------------------
The following example shows how to train a model using our own training engine.
We provide a `Trainer` class that handles the training and evaluation loop for you.
For more details, head over to the `Trainer <TODO>`_ class documentation.

.. code-block:: python

   from vis4d.engine.experiment import run_experiment
   from vis4d.config import instantiate_classes
   from vis4d.engine.optim import set_up_optimizers

   # Load your Config here
   # from your_config import get_config
   config = get_config()
   model = instantiate_classes(config.model)

   # Callbacks
   callbacks = [instantiate_classes(cb) for cb in config.callbacks]
   mode = "fit|test" # Set to "fit" if you want to train a model, "test" if you want to evaluate a model

    # Setup Dataloaders & seed
    if mode == "fit":
        train_dataloader = instantiate_classes(config.data.train_dataloader)
        train_data_connector = instantiate_classes(config.train_data_connector)
        optimizers, lr_schedulers = set_up_optimizers(config.optimizers, [model])
        loss = instantiate_classes(config.loss)
    else:
        train_dataloader = None
        train_data_connector = None

    test_dataloader = instantiate_classes(config.data.test_dataloader)
    test_data_connector = instantiate_classes(config.test_data_connector)

    trainer = Trainer(
        device=device,
        output_dir=config.output_dir,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_data_connector=train_data_connector,
        test_data_connector=test_data_connector,
        callbacks=callbacks,
        num_epochs=config.params.get("num_epochs", -1),
    )

    if mode == "fit":
        trainer.fit(model, optimizers, lr_schedulers, loss)
    elif mode == "test":
        trainer.test(model)