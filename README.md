# SYSTM

A perception system of tracking and motion understanding.

## Installation

We currently support Python 3.7 and 3.8.

You can install the package dependency via vanilla python and pip:

```bash
python3 -m pip install -r scripts/requirements.txt
# This may not work for you. Please look up pytorch website for your configurations
python3 -m pip install torch torchvision torchaudio
# Install fresh detectron2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you're using conda, run the following commands:

```bash
conda create --name systm python=3.8
conda activate systm
pip install -r scripts/requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

You can also use [python](./scripts/setup_linux_cpu.sh) and [conda](./scripts/setup_linux_cpu_conda.sh) scripts on CPU Linux installation.

More information about torch and detectron2 installation

- [Pytorch](https://pytorch.org/get-started/locally)
- [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

To install `systm` package,

```bash
python3 setup.py install
```

## Usage

Training detector

```bash
python3 tools/detect.py train --config <config_path> <maybe other arguments>
```

Generate detection prediction results

```bash
python3 tools/detect.py predict --config <config_path> <maybe other arguments>
```
