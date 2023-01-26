Installation
===============================================

We currently support Python 3.9+.

First, create a new virtual environment, e.g. with conda:

.. code:: bash

    conda create --name vis4d python=3.10
    conda activate vis4d

Next, install the library and its dependencies:

.. code:: bash

    pip install --ignore-installed -r requirements.txt
    python setup.py install

If you want to install torch with gpu:

.. code:: bash

    pip install --ignore-installed -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
    python setup.py install

More information about torch and pytorch-lightning installation

- [Pytorch](https://pytorch.org/get-started/locally)
- [PytorchLightning](https://www.pytorchlightning.ai/)

The standard location for datasets used in the experiments is:

.. code:: bash

    --root
        --data
            --dataset1
            --dataset2

and the original folder structure of the data.

If you don’t want to modify the package:

.. code:: bash

    python setup.py install


If you want to modify the package:

.. code:: bash

    python setup.py develop

See `here <https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install>`_ for further explanation.
For systems where you cannot write to the python installation use:

.. code:: bash

    python setup.py develop --user
