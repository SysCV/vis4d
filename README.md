# SYSTM

A perception system of tracking and motion understanding.

## Install dependencies

We currently support Python 3.7 and 3.8.

- Install the pip dependencies:

  ```bash
  pip3 install -r requirements.txt
  ```

- [Install Pytorch](https://pytorch.org/get-started/locally)
- [Install Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)


Example setup with conda and python 3.8:
```
conda create --env systm python=3.8
conda activate systm
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
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
