## Vis4D Documentation

**Note**

- All images and videos should go to img/ folder
- The documentation will be automatically deployed to the website once a PR is merged to main

### Build

Install dependencies

```
pip install -r requirements.txt
```

Build the full documentation

```
make html
```

_Please make sure the documentation can build correctly before merging changes to main!_

The full doc files will be in `build` and can be displayed in the browser by opening `build/html/index.html`.

## Setting up and Training your Model

We provide our config system as well as our own engine and command line interface. This allows us to train models in a very flexible way. We also provide a number of pre-defined models and datasets that you can use to train your models.

### Config System

The config system is a simple way to configure your model. It heavily relies on the [ml_collections](https://github.com/google/ml_collections) library and is structured in a similar way.
In order to configure your model and training pipeline, a python file with a `get_config` function is required. This function should return a `ConfigDict` object. Have a look at the example config files located at `vis4d/config/examples` for more information.

Additionally, you can also define a `get_sweep` function that returns a `ConfigDict` object in the same file. This function is used to define a hyperparameter sweep.

### Training

To train a model, you can either use our engine or use the pytorch_lightning trainer directly. The CLI interfaces for both are the same, however, the main script is either `vis4d.engine.cli` or `vis4d.pl.cli`.

Here are a few examples on how to train a model using the CLI interfaces:

#### Using our engine

```bash
# Training faster_rcnn on the coco dataset and one gpu
python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py --gpus 1
# You can overwrite any config value by passing it to the cli. This updates the "num_epochs" value in the config dict.
python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py --gpus 1 --config.num_epochs 100
# You can also perform a hyperparameter sweep by passing a sweep config file to the cli.
python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py --sweep vis4d/config/example/faster_rcnn_coco.py
```

#### Using Pytorch Lightning

```bash
# Training faster_rcnn on the coco dataset and one gpu
python -m vis4d.pl.cli --config vis4d/config/example/faster_rcnn_coco.py --gpus 1
```
