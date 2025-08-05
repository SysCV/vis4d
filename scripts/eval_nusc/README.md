# nuScenes 3D Detection and Tracking Evaluation

This folder contains the code and python environment to run nuScenes 3D detection and tracking evaluation locally.

### Installation
- Python: 3.6

```bash
pip install -r nusc.txt
```

### Run
```bash
# Detection
python run.py \
--input $FOLDER_OF_PREDICTION \
--version $VERSION \
--dataroot $NUSC_DATA_ROOT \
--mode detection

# Tracking
python run.py \
--input $FOLDER_OF_PREDICTION \
--version $VERSION \
--dataroot $NUSC_DATA_ROOT \
--mode tracking
```