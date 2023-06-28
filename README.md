# Vis4D

A library for dynamic scene understanding.

## Installation

Installation is as easy as:

```bash
python3 -m pip install .
```

[For more detailed information, check out our installation guide](docs/source/tutorials/install.rst)

## Basic CLI usage

- To train a model, e.g. Faster-RCNN on COCO

```bash
# vis4d.engine
python -m vis4d fit --config configs/faster_rcnn/faster_rcnn_coco.py --gpus 1

# vis4d.pl
python -m vis4d.pl fit --config configs/faster_rcnn/faster_rcnn_coco.py --gpus 1
```

- To test a model

```bash
# vis4d.engine
python -m vis4d test --config configs/faster_rcnn/faster_rcnn_coco.py --gpus 1

# vis4d.pl
python -m vis4d.pl test --config configs/faster_rcnn/faster_rcnn_coco.py --gpus 1
```

## DDP

### Training

- Local machine / SLURM interactivate job (`job-name=bash`)

```bash
# vis4d.engine
./scripts/dist_train.sh <config-file> <num-gpus>

# vis4d.pl
python -m vis4d.pl fit --config <config-file> --gpus <num-gpus>
```

- SLURM batch job. Need to config the submission file.

```bash
# vis4d.engine
sbatch scripts/slurm_train.sh

# vis4d.pl
srun --cpus-per-task=4 --gres=gpumem:20G python -m vis4d.pl.run fit \
    --config <config-file> --gpus <num-gpus>
```

### Testing

- Local machine / SLURM interactivate job (`job-name=bash`)

```bash
# vis4d.engine
./scripts/dist_test.sh <config-file> <num-gpus>

# vis4d.pl
python -m vis4d.pl test --config <config-file> --gpus <num-gpus>
```

- SLURM batch job. Need to config the submission file.

```bash
# vis4d.engine
sbatch scripts/slurm_test.sh

# vis4d.pl
srun --cpus-per-task=4 --gres=gpumem:20G python -m vis4d.pl.run test \
    --config <config-file> --gpus <num-gpus>
```

## Contribute

[Check out our contribution guidelines for this project](docs/source/contribute.rst)
