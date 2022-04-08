# Quasi-Dense Similarity Learning for Multiple Object Tracking


This is the updated official implementation of the paper [Quasi-Dense Similarity Learning for Multiple Object Tracking](https://arxiv.org/abs/2006.06664).

## Model Zoo

We provide pre-trained models for all main experiments:

### BDD100K

To reproduce our training, run:
```python
 python projects/qdtrack/run.py fit --data.experiment bdd100k --trainer.gpus 8 --data.samples_per_gpu 2
 ```

Note that we train our model on 8 RTX 2080 Ti GPUs. If you use a different configuration, adjust the parameters accordingly.

##### TODO update numbers, model link
| mMOTA | mIDF1  | ID Sw. |
|-------|--------|--------|
| 35.5  | 52.3   |  10790 |

To test the pretrained model, run:
```python
 python projects/qdtrack/run.py test --data.experiment bdd100k --trainer.gpus <number of available gpus> --ckpt_path <path to weights>
 ```

### MOT

##### TODO update numbers, model link
| Dataset | MOTA | IDF1  | ID Sw. | MT | ML |
|-------|--------|--------| ----| ---| ---|
| MOT17 | 69.8 | 67.1 | 1097 | 316 | 150 |
| MOT20 | 68.7 | 66.3 | 3378 | 957 | 516 |


### DanceTrack