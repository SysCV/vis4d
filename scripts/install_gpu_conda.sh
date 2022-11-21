#!/bin/bash
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
echo https://download.pytorch.org/whl/cu$1/torch_stable.html
echo conda install cudatoolkit=$1 -c pytorch
# conda create --name vis4d python=3.10
# conda activate vis4d-dev
# conda install cudatoolkit=$1 -c pytorch
# python3 -m pip install --ignore-installed  -r requiremets/base.txt -f https://download.pytorch.org/whl/cu${$1//.}/torch_stable.html
# python3 setup.py develop