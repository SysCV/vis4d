conda create --name systm python=3.8
conda activate systm
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
python -m pip install --ignore-installed  -r scripts/requirements.txt
python setup.py install