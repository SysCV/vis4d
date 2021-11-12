# Vis4D

A framework for dynamic scene understanding.

## Installation

We currently support Python 3.7, 3.8 and 3.9.

### Minimal requirements
You can install the minimal package dependencies via vanilla python and pip:

```bash
python3 -m pip install -r requirements.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

This command installs pytorch without CUDA. Please look up
[pytorch website](https://pytorch.org/get-started/locally) for installation
on your configurations and install pytorch first.

If you're using conda, run the following commands:

```bash
conda create --name vis4d python=3.8
conda activate vis4d
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

More information about torch and pytorch-lightning installation

- [Pytorch](https://pytorch.org/get-started/locally)
- [PytorchLightning](https://www.pytorchlightning.ai/)

### Full requirements
To install the full package dependencies, please refer to the [python](scripts/install_cpu_dep_full.sh) and [conda](scripts/install_cpu_conda_dep_full.sh) scripts.
Or install the following packages:

- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmdet](https://github.com/open-mmlab/mmdetection)
- [mmseg](https://github.com/open-mmlab/mmsegmentation)
- [detectron2](https://github.com/facebookresearch/detectron2)

Please note that specific models will require some of these packages.

### Package installation
To install `vis4d` package,

```bash
python3 setup.py install
```

## Usage

Training

```bash
python3 -m vis4d.engine.trainer train --config <config_path> <maybe other arguments>
```

Testing

```bash
python3 -m vis4d.engine.trainer test --config <config_path> <maybe other arguments>
```

Prediction

```bash
python3 -m vis4d.engine.trainer predict --config <config_path> <maybe other arguments>
```
