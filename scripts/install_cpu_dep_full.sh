#!/bin/bash
python3 -m pip install -r requirements/install.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
python3 -m pip install -r requirements/torch-lib.txt
python3 -m pip install -r requirements/viewer.txt
python3 -m pip install -r requirements/dev.txt
