# VisT

A perception system of tracking and motion understanding.

## Installation

We currently support Python 3.7, 3.8 and 3.9.

You can install the package dependency via vanilla python and pip:

```bash
python3 -m pip install -r requirements.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

This command installs pytorch without CUDA. Please look up
[pytorch website](https://pytorch.org/get-started/locally) for installation
on your configurations and install pytorch first.

If you're using conda, run the following commands:

```bash
conda create --name vist python=3.8
conda activate vist
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

You can also use [python](./scripts/install_cpu_dep.sh) and [conda](./scripts/install_cpu_conda_dep.sh) scripts on CPU machine installation.

More information about torch and pytorch-lightning installation

- [Pytorch](https://pytorch.org/get-started/locally)
- [PytorchLightning](https://www.pytorchlightning.ai/)

To install `vist` package,

```bash
python3 setup.py install
```

## Usage

Training

```bash
python3 -m vist.engine.trainer train --config <config_path> <maybe other arguments>
```

Testing

```bash
python3 -m vist.engine.trainer test --config <config_path> <maybe other arguments>
```

Prediction

```bash
python3 -m vist.engine.trainer predict --config <config_path> <maybe other arguments>
```
