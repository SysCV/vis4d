# CC-3DT++: The Velocity-Similarity Enhanced Tracker of CR3DT
This packages contains the isolated CC-3DT++ tracker of [CR3DT](https://arxiv.org/abs/2403.15313) and is meant to be run in combination with the [CR3DT](https://github.com/ETH-PBL/CR3DT) detector.

## Installation

Step 1: Clone the repository
```bash
git clone https://github.com/ETH-PBL/cc-3dt-pp.git
cd cc-3dt-pp
```
Step 2: Build the docker image
```bash
docker build -t cc3dt -f Dockerfile.cc3dtpp .
```

Step 3. Create the folder structure below anywhere on your file system. You can chose to populate the folders with the nuScenes dataset and our provided pkl-files and checkpoints ([Google Drive with checkpoint and pkls](https://drive.google.com/drive/folders/1gHPZMUCDObDTHqbU_7Drw0CILx4pu_7i)), or just with the dataset and to create any pkl-files and checkpoints yourself. At least one of the three dataset folders (`v1.0-mini`, `v1.0-trainval`, or `v1.0-test`) needs to be populated.
```shell script
...
├── <your data directory>
│   ├── v1.0-mini
│   ├── v1.0-trainval
│   ├── v1.0-test
│   └── checkpoints
└ ...
```

Step 4. Start the docker container with the necessary flags using the provided utility script. After that you can open a second interactive shell to the docker using `sec_docker.sh`.
```shell script
./main_cc3dtpp_track.sh <path to your data directory>
./sec_docker.sh
```

Step 5. Make sure you have gnerated a results json file for the nuScenes dataset. You can use the following command to generate the results json file and place it in the corresponding datasets folder. Or you can download our provided detection results [here](https://drive.google.com/drive/folders/1gHPZMUCDObDTHqbU_7Drw0CILx4pu_7i).

Step 6. Inside the docker container, you can run the following command to evaluate the tracking performance on nuscense dataset. This will mount the datasets into the container in the correct naming convention.
```shell script
# For mini
./full_eval.sh mini vis4d/data/nuscenes_mini/<results json file>

# For trainval
./full_eval.sh trainval vis4d/data/nuscenes_trainval/<results json file>

# For test
./full_eval.sh test vis4d/data/nuscenes_test/<results json file>
```

## Expected Performance

### CC-3DT++ On nuScenes trainval
```
AMOTA   0.381
AMOTP   1.366
RECALL  0.461
MOTAR   0.729
GT      14556
MOTA    0.342
MOTP    0.711
MT      2120
ML      2505
FAF     38.5
TP      55658
FP      10808
FN      45034
IDS     1205
FRAG    2012
TID     1.44
LGD     2.52
```

### CC-3DT++ On nuScenes mini
```
AMOTA   0.477
AMOTP   1.226
RECALL  0.587
MOTAR   0.803
GT      611
MOTA    0.476
MOTP    0.703
MT      62
ML      53
FAF     96.2
TP      2132
FP      413
FN      1437
IDS     100
FRAG    124
TID     1.42
LGD     5.02
```

## Acknowledgement
For CC-3DT++ we build upon the original Vis4D codebase, which is a group effort by our team at ETH Zurich. We would like to thank the authors for their contribution to the open-source community.
Vis4D is a group effort by our team at ETH Zurich.

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

If you find CC-3DT++ is useful for your research, please consider citing the following BibTeX entry.

```bibtex
@article{baumann2024cr3dt,
  title={CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking},
  author={Baumann, Nicolas and Baumgartner, Michael and Ghignone, Edoardo and K{\"u}hne, Jonas and Fischer, Tobias and Yang, Yung-Hsu and Pollefeys, Marc and Magno, Michele},
  journal={arXiv preprint arXiv:2403.15313},
  year={2024}
}
```
