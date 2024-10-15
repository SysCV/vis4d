***************
Troubleshooting
***************

-----------
File limits
-----------

.. code:: bash

    RuntimeError: received 0 items of ancdata

Please check `ulimit -n` on your machine and if it is of value 1024 or lower, try increasing it to 2048 or 4096. See https://github.com/pytorch/pytorch/issues/973 for further info.

-----------------
CPU out of memory
-----------------

.. code:: bash

    OSError: [Errno 12] Cannot allocate memory

Try setting `workers_per_gpu` to a lower value (usually 1/2 of `samples_per_gpu` is sufficient).
