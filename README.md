# SYSTM

A perception system of tracking and motion understanding.

## Installation

We currently support Python 3.7 and 3.8. For vanilla python, please first install:

- [Pytorch](https://pytorch.org/get-started/locally)
- [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

Next, you can install the package via pip:

```bash
pip install -r scripts/requirements.txt
python setup.py install
```


If you're using conda, run the following commands:
```
conda create --name systm python=3.8
conda activate systm
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r scripts/requirements.txt
python setup.py install
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
