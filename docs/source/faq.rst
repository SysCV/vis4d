FAQ
=====


The startup takes too long
++++++++++++++++++++++++++++

Activate the 'cache_as_binary' option in your datasets. This will create a binary file that stores the annotations, which is faster to load.


Key and reference frames
++++++++++++++++++++++++++++

While the data pipeline gives you the option to order your input frames sequentially, one frame will always contain the attribute keyframe, meaning that this is the frame that has been selected from the dataset, while the others were sampled according to the reference view sampling parameters. This is important, since the keyframe ensures that each frame in the dataset is sampled once per epoch, while the reference views are chosen randomly.


Troubleshooting
==================
Training crashes with:

.. code:: bash

    RuntimeError: received 0 items of ancdata

Please check `ulimit -n` on your machine and if it is of value 1024 or lower, try increasing it to 2048 or 4096. See https://github.com/pytorch/pytorch/issues/973 for further info.

.. code:: bash

    OSError: [Errno 12] Cannot allocate memory

Try setting `workers_per_gpu` to a lower value (usually 1/2 of `samples_per_gpu` is sufficient).

