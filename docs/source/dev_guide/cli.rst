====
CLI
====
We provide a command line interface for training and evaluating your models.
Assuming you have installed the package using pip, you can use the command `vis4d` to access the CLI.

Alternatively, you can run the CLI using `python -m vis4d.engine.cli` or `python -m vis4d.pl.cli` if you want to use the PyTorch Lightning version.

The CLI relies on a configuration file to specify each experiment. We use `ml_collections <https://github.com/google/ml_collections>`_ as underlying framework to define the configuration files.
You can read up on our configuration files in the `Config System <configuration_files>`_ section.

-------------
CLI Interface
-------------
The provided examples assume that the experiment configuration file is located at `path_to_experiment_cfg.py`.
You can read up on our configuration files in the `Config System <configuration_files>`_ section.

We support both, our own training engine as well as `PyTorch Lightning <https://www.pytorchlightning.ai/>`_.

^^^^^^^^^^^^
CLI Commands
^^^^^^^^^^^^
.. code-block:: bash

  vis4d [train|test] --config path_to_experiment_cfg.py
                     --ckpt: Checkpoint path
                     --config: path to config file [.py |.yaml].
                     --gpus: Number of GPUs to use. Default 0.
                     --print-config: If set, prints the configuration to the console.
                     --resume: Resume training from the provided checkpoint.
                     --sweep: path to a parameter sweep configuration file [.py |.yaml].

.. code-block:: bash

  vis4d-pl [train|test] --config path_to_experiment_cfg.py
                     --ckpt: Checkpoint path
                     --config: path to the config file [.py |.yaml].
                     --gpus: Number of GPUs to use. Default 0.
                     --print-config: If set, prints the configuration to the console.
                     --resume: Resume training from the provided checkpoint.

Quick Start
^^^^^^^^^^^^^

**Train a model**

.. code-block:: bash

  vis4d fit --config path_to_experiment_cfg.py


**Test a model**

.. code-block:: bash

   vis4d fit --config path_to_experiment_cfg.py

**Overwrite Config Parameters**

.. code-block:: bash

   vis4d fit --config path_to_experiment_cfg.py --config.my_field=2 --config.my.nested.field="test"



**Perform Parameter Sweeps**

.. code-block:: bash

   vis4d fit --config path_to_experiment_cfg --sweep path_to_sweep_cfg.py

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Overwrite Config Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We support overwriting config parameters via the CLI. Assuming you have a config parameter `params.lr` in your config file, you can overwrite it using the following command:

.. code-block:: bash

   vis4d train --config path_to_experiment_cfg.py --config.params.lr=0.01

Note that misstyping a config parameter

.. code-block:: bash

   vis4d train --config path_to_experiment_cfg.py --config.params.lrs=0.01

will result in the following error:
.. code-block:: bash

   AttributeError: Did you mean "lr" instead of "lrw"?'


Callbacks and Trainer Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We support custom Callbacks as well as Pytorch Lightning Trainer Arguments.
Head over to the `Config System <configuration_files>`_ section to learn more about how to use them.

Using the Python API
--------------------
While we provide a CLI for training and evaluating your models, you can also use the python API directly.

Using the Trainer class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
