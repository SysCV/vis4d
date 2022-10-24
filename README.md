# Vis4D

A library for dynamic scene understanding.

## Installation

Installation is as easy as:

```bash
python3 -m pip install .
```

[For more detailed information, check out our installation guide](docs/source/tutorials/install.rst)


## Basic CLI usage

To {fit,validate,test,predict,tune} a model, e.g. Faster-RCNN, run:

```bash
python -m vis4d.pl.model.faster_rcnn {fit,validate,test,predict,tune} --data.experiment coco --trainer.gpus 1
```

To check the command line arguments, run:

```bash
python -m vis4d.pl.model.faster_rcnn fit -h
```

## Contribute

[Check out our contribution guidelines for this project](docs/source/contribute.rst)

