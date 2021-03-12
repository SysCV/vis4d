# OpenMT

A perception system of tracking and motion understanding.

## Installation

We currently support Python 3.7 and 3.8.

You can install the package dependency via vanilla python and pip:

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

This command install pytorch without CUDA. Please look up pytorch website for your configurations.

If you're using conda, run the following commands:

```bash
conda create --name openmt python=3.8
conda activate openmt
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

You can also use [python](./scripts/install_cpu_dep.sh) and [conda](./scripts/setup_linux_cpu_conda.sh) scripts on CPU machine installation.

More information about torch and detectron2 installation

- [Pytorch](https://pytorch.org/get-started/locally)
- [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

To install `openmt` package,

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
