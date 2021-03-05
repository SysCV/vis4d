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

## Usage

Training detector

```bash
python3 tools/detect.py train --config <config_path> <maybe other arguments>
```

Generate detection prediction results

```bash
python3 tools/detect.py predict --config <config_path> <maybe other arguments>
```
