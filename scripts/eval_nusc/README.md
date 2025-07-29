# nuScenes 3D Detection and Tracking Evaluation

This folder contains the code and python environment to run nuScenes 3D detection and tracking evaluation locally.

### Installation
- Python: 3.6

```bash
pip install -r nusc.txt
```

### Run
- $WORK_DIR is your output folder which contains the prediction json file.
- $VERSION is `mini` or `trainval` to select mini or validation split.

```bash
bash eval_nusc.sh $WORK_DIR $VERSION
```