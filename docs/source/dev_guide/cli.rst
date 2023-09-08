###
CLI
###
We provide a command line interface for training and evaluating your models.
Assuming you have installed the package using pip, you can use the command `vis4d` to access the CLI.

Alternatively, you can run the CLI using `python -m vis4d.engine.cli` or `python -m vis4d.pl.cli` if you want to use the PyTorch Lightning version.

The CLI relies on a configuration file to specify each experiment. We use `ml_collections <https://github.com/google/ml_collections>`_ as underlying framework to define the configuration files.
You can read up on our configuration files in the `Config System <configuration_files>`_ section.

=============
CLI Interface
=============
The provided examples assume that the experiment configuration file is located at `path_to_experiment_cfg.py`.
You can read up on our configuration files in the `Config System <configuration_files>`_ section.

We support both, our own training engine as well as `PyTorch Lightning <https://www.pytorchlightning.ai/>`_.

------------
CLI Commands
------------
.. code-block:: bash

  vis4d [fit | test] --config path_to_experiment_cfg.py
                     --ckpt: Checkpoint path
                     --config: path to config file [.py |.yaml].
                     --gpus: Number of GPUs to use. Default 0.
                     --print-config: If set, prints the configuration to the console.
                     --resume: Resume training from the provided checkpoint.
                     --sweep: path to a parameter sweep configuration file [.py |.yaml].

.. code-block:: bash

  vis4d-pl [fit | test] --config path_to_experiment_cfg.py
                        --ckpt: Checkpoint path
                        --config: path to the config file [.py |.yaml].
                        --gpus: Number of GPUs to use. Default 0.
                        --print-config: If set, prints the configuration to the console.
                        --resume: Resume training from the provided checkpoint.

-----------
Quick Start
-----------

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

---------------------------
Overwrite Config Parameters
---------------------------

We support overwriting config parameters via the CLI. Assuming you have a config parameter `params.lr` in your config file, you can overwrite it using the following command:

.. code-block:: bash

   vis4d fit --config path_to_experiment_cfg.py --config.params.lr=0.01

Note that misstyping a config parameter

.. code-block:: bash

   vis4d fit --config path_to_experiment_cfg.py --config.params.lrs=0.01

will result in the following error:
.. code-block:: bash

   AttributeError: Did you mean "lr" instead of "lrw"?'
