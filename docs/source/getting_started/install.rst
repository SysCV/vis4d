Installation
===============================================
First, create a new virtual environment, e.g. with conda:

.. code:: bash

    conda create --name vis4d python=3.8
    conda activate vis4d
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
    pip install --ignore-installed  -r requirements.txt
    pip install mmcv-full
    pip install git+https://github.com/SysCV/mmdetection.git


Optionally, detectron2 and mmsegmentation can be installed with:


.. code:: bash

    python3 -m pip install git+https://github.com/facebookresearch/detectron2.git
    python3 -m pip install mmsegmentation


The standard location for datasets in the experiments in configs/experiments is:

.. code:: bash

    --data
        --bdd100k
        --MOT17
    --train
        --test
    --nuscenes

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


Common issues
+++++++++++++++++++++++++++++++++++++++

Error: “Expected 88 from C header, got 80 from PyObject”

.. code:: bash

    pip uninstall numpy; pip install numpy

This can happen when your pycocotools version is not compatible with your numpy version. More info here.

When running on Euler using CVL settings, use this instead:

.. code:: bash

    pip install --ignore-installed numpy

