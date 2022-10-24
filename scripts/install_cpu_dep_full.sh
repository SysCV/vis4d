#!/bin/bash
python3 -m pip install -r base.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install -r optional.txt
python3 -m pip install -r dev.txt
python3 -m pip install -r visualization.txt
