Customize
===========
In the following we will explain how to customize Vis4D based on your needs.

Implement a custom dataset
+++++++++++++++++++++++++++++

To implement a custom dataset, you need to specify a class that loads your annotations and converts them to Scalabel format. This is done via inheriting from the `vis4d.data.datasets.BaseDatasetLoader` class, that will require you to implement the `load_dataset` function.
You can define additional arguments by overwriting the `__init__` function of `BaseDatasetLoader`.

.. code:: python

    from vis4d.data.datasets.base import BaseDatasetLoader

    class MyDataset(BaseDatasetLoader):

        def __init__(self, *args, some_positional_arg, some_optional_arg = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.some_positional_arg = det_metrics_per_video
            self.some_optional_arg = some_optional_arg

        def load_dataset(self) -> Dataset:
            # load or convert your annotations into scalabel format
            dataset = ...
            return dataset

        def evaluate(self, metric: str, predictions: List[Frame], gts: List[Frame]) -> Tuple[MetricLogs, str]:
            # Optionally, you can define evaluation functions beyond the standard
            # metrics in scalabel, e.g.:
            if metric == "my_fancy_metric":
                metrics, log_string = my_fancy_eval(predictions, gts)
            else:
                metrics, log_string = super().evaluate(metric, predictions, gts)
            return metrics, log_string


Implement a custom model
+++++++++++++++++++++++++++++

To define a new model, just define a model class that inherits from `vis4d.op.BaseModel`.
The model usually should implement the following two functions:

- `forward_train`: execute the model on the inputs, compute losses, return losses
- `forward_test`: run model inference on given inputs, return predictions

Note that while we define the standard behavior via `forward_train` and `forward_test`, since `vis4d.op.BaseModel` inherits from `pytorch_lightning.LightningModule`, all standard behavior can be modified by overwriting the respective function of `LightningModule`, such as `training_step`, `test_step`, `optimizer_step`, etc.
Furthermore, a variety of callbacks can be added to define additional functionality at certain points during execution. For further information, see the documentation of `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html>`_ and `Callbacks <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_

.. code:: python

    from vis4d.op import BaseModel

    class MyModel(BaseModel):

        def __init__(self, *args, some_positional_arg, some_optional_arg = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.some_positional_arg = det_metrics_per_video
            self.some_optional_arg = some_optional_arg

        def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
            # implement the forward pass during training, return a dict with
            # all losses you want to log / backprop

        def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
            # implement the forward pass during testing, return a dict with
            # Scalabel format predictions for all inputs


You can find more detailed customization examples in the `projects` folder.