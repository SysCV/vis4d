#!/bin/bash
eval "$(conda shell.bash hook)"
conda create --name openmt python=3.8
conda activate openmt
python3 -m pip install --ignore-installed  -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch -c nvidia
# python -m pip install detectron2 -f \
#   https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
