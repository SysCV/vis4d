############
Installation
############

=======
Package
=======
We currently support Python 3.9+ and PyTorch 1.10.0+.

We recommand to install in a new virtual environment, e.g. conda or virtualenv.

-----------------
Install with PyPI
-----------------

You can install the library as easy as

.. code:: bash

    python3 -m pip install vis4d

-----------------
Build from source
-----------------

If you want to build the package from source and specify CUDA version, you can clone the repository and install it:

.. code:: bash

    git clone https://github.com/SysCV/vis4d.git
    cd vis4d

    python3 -m pip install -r requirements/install.txt -f https://download.pytorch.org/whl/cu117/torch_stable.html
    python3 -m pip install -r requirements/torch-lib.txt
    python3 -m pip install -e .

More information about torch and pytorch-lightning installation

- `PyTorch <https://pytorch.org/get-started/locally>`_
- `PyTorch Lightning <https://lightning.ai/docs/pytorch/latest/>`_

================
Directory Layout
================
You can use the library in different folder structures and codebase.
But by default Vis4D will use the following directories by default:

----
Data
----
The default location for datasets used in the experiments is:

.. code:: bash

    --root
        --data
            --dataset1
            --dataset2

---------
Workspace
---------
The default output folder used in the experiments to store logs, checkpoints, etc. is:

.. code:: bash

    --root
        --vis4d-workspace
            --experiment_name1
                --version1
                --version2
            --experiment_name2
                --version1
                --version2
