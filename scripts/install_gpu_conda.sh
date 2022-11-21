#!/bin/bash
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh

if [ -z "$1" ]
then
    CUDA_VERSION=11.7
else
    CUDA_VERSION=$1
fi

conda create --name vis4d python=3.10
conda activate vis4d-dev
conda install cudatoolkit=$CUDA_VERSION -c pytorch -c nvidia
python3 -m pip install --ignore-installed  -r requirements/base.txt -f https://download.pytorch.org/whl/cu${CUDA_VERSION//.}/torch_stable.html
python3 setup.py develop
