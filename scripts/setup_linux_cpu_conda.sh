eval "$(conda shell.bash hook)"
conda create --name openmt python=3.8
conda activate openmt
python -m pip install --ignore-installed  -r scripts/requirements.txt
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'