Customize
===========
Vis4D can be customized very easily.

Implement a custom dataset
+++++++++++++++++++++++++++++

To implement a custom dataset, you need to specify a class that loads your annotations and converts them to Scalabel format. This is done via inheriting from the `vis4d.data.datasets.BaseDatasetLoader` class, that will require you to implement the `load_dataset` function.
You can define additional arguments by overwriting the `__init__` function of `BaseDatasetLoader`. See e.g. `vis4d.data.datasets.waymo` for an example of how to extend the configuration.


Implement a custom model
+++++++++++++++++++++++++++++

To define a new model, just define a model class that inherits from `vis4d.model.BaseModel`.
The model usually should implement the following two functions:
- `forward_train`: execute the model on the inputs, compute losses, return losses
- `forward_test`: run model inference on given inputs, return predictions

Note that while we define the standard behavior via `forward_train` and `forward_test`, since `vis4d.model.BaseModel` inherits from `pytorch_lightning.LightningModule`, all standard behavior can be modified by overwriting the respective function of `LightningModule`, such as `training_step`, `test_step`, `optimizer_step`, etc.
Furthermore, a variety of callbacks can be added to define additional functionality at certain points during execution. For further information, see the documentation of `LightningModule<https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html>`_ and `Callbacks<https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_

You can also find customization examples in the `projects` folder.