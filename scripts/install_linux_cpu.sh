python3 -m pip install --ignore-installed -r scripts/requirements.txt
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu \
    torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
