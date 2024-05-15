<p align="center">
    <!-- pypi-strip -->
    <picture>
    <!-- /pypi-strip -->
    <img alt="vis4d" src="https://dl.cv.ethz.ch/vis4d/vis4d_logo.svg" width="400">
    <!-- pypi-strip -->
    </picture>
    <!-- /pypi-strip -->
    <br/>
    A modular library for 4D scene understanding
</p>

## Quickstart

You can checkout our [documentation](https://docs.vis.xyz/4d/index.html).

You can use the [template](https://github.com/SysCV/vis4d-template) here to start your own project with Vis4D.

## Installation

Installation is as easy as

```bash
python3 -m pip install vis4d
```

[For more detailed information, check out our installation guide](docs/source/user_guide/install.rst)

## Basic CLI usage

- To train a model, e.g. Faster-RCNN on COCO

```bash
# vis4d.engine
vis4d fit --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py --gpus 1

# vis4d.pl
vis4d-pl fit --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py --gpus 1
```

- To test a model

```bash
# vis4d.engine
vis4d test --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py --gpus 1

# vis4d.pl
vis4d-pl test --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py --gpus 1
```

## DDP

### Training

- Local machine / SLURM interactivate job (`job-name=bash`)

```bash
# vis4d.engine
./scripts/dist_train.sh <config-file> <num-gpus>

# vis4d.pl
vis4d-pl fit --config <config-file> --gpus <num-gpus>
```

- SLURM

```bash
# vis4d.engine
srun vis4d fit --config <config-file> --gpus <num-gpus> --slurm True

# vis4d.pl
srun vis4d-pl fit --config <config-file> --gpus <num-gpus>
```

### Testing

- Local machine / SLURM interactivate job (`job-name=bash`)

```bash
# vis4d.engine
./scripts/dist_test.sh <config-file> <num-gpus>

# vis4d.pl
vis4d-pl test --config <config-file> --gpus <num-gpus>
```

- SLURM

```bash
# vis4d.engine
srun vis4d test --config <config-file> --gpus <num-gpus> --slurm True

# vis4d.pl
srun vis4d-pl test --config <config-file> --gpus <num-gpus>
```

## Acknowledgement
Vis4D is a group effort by our team at ETH Zurich.
[Yung-Hsu Yang](https://royyang0714.github.io/) built the current version and will be the main maintainer of the codebase.

Vis4D was originally written by [Tobias Fischer](https://tobiasfshr.github.io/) during the first three years of his Ph.D. at ETH Zurich, [Thomas E. Huang](https://www.thomasehuang.com/) helped contribute many models, [Tao Sun](https://www.suniique.com/) implemented the ViT models and designed the evaluation pipeline, and[René Zurbrügg](https://github.com/renezurbruegg) designed the config system.


## Contributors
**Project Leads**
- [Yung-Hsu Yang](https://royyang0714.github.io/)*
- [Tobias Fischer](https://tobiasfshr.github.io/)*
 
**Core Contributors**
- [Thomas E. Huang](https://www.thomasehuang.com/)
- [Tao Sun](https://www.suniique.com/)
- [René Zurbrügg](https://renezurbruegg.github.io/)
 
**Advisors**
- [Fisher Yu](https://www.yf.io/)
 
`*` denotes equal contribution

**We are open to contributions and suggestions, feel free to reach out to us.**

[Check out our contribution guidelines for this project](docs/source/dev_guide/contribute.rst)

**Community Contributors**
 
<a href="https://github.com/SysCV/vis4d/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SysCV/vis4d" />
</a>


## Citation

If you find Vis4D is useful for your research, please consider citing the following BibTeX entry.

```bibtex
@misc{vis4d_2024,
  author = {{Yung-Hsu Yang and Tobias Fischer and Thomas E. Huang} and René Zurbrügg and Tao Sun and Fisher Yu},
  title = {Vis4D},
  howpublished = {\url{https://github.com/SysCV/vis4d}},
  year = {2024}
}
```
