# Installation

We currently support Python 3.11+ and PyTorch 2.4.0+.

We recommand to install in a new virtual environment, e.g. conda or virtualenv.

## PyPI

You can install the library as easy as

```bash
python3 -m pip install vis4d
```


## Build from source

If you want to build the package from source, you can clone the repository and install it:

```bash
git clone https://github.com/SysCV/vis4d.git
cd vis4d

python3 -m pip install -e .
```

More information about torch and pytorch-lightning installation

- [PyTorch](https://pytorch.org/get-started/locally)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)


## CUDA Operations

Some functionalities in the library require CUDA operations. You can install them by running:

```bash
python3 -m pip install -r requirements/torch-lib.txt
```
