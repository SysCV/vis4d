=====
CLI
=====
We provide a command line interface for training and evaluating your models.

TL;DR
=====
Train a model
------
.. code-block:: bash

   python -m vis4d.engine.cli train --config <path_to_experiment_cfg>


Test a model
------
.. code-block:: bash

   python -m vis4d.engine.cli test --config <path_to_experiment_cfg>

Parameter Sweeps
------

.. code-block:: bash

   python -m vis4d.engine.cli train --config <path_to_experiment_cfg> --sweep <path_to_sweep_cfg>


Overwrite Config Parameters
------

.. code-block:: bash

   python -m vis4d.engine.cli train --config <path_to_experiment_cfg> --config.my_field=2 --config.my.nested.field="test"
