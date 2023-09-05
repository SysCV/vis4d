****
Data
****

==========================
The startup takes too long
==========================

Activate the 'cache_as_binary' option in your datasets. This will create a binary file that stores the annotations, which is faster to load.

========================
Key and reference frames
========================

While the data pipeline gives you the option to order your input frames sequentially, one frame will always contain the attribute keyframe, meaning that this is the frame that has been selected from the dataset, while the others were sampled according to the reference view sampling parameters. This is important, since the keyframe ensures that each frame in the dataset is sampled once per epoch, while the reference views are chosen randomly.
