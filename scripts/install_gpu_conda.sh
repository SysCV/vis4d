CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda create --name vis4d python=3.8
conda activate vis4d
conda install cudatoolkit=11.3 -c pytorch
python3 -m pip install --ignore-installed  -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
python3 setup.py develop