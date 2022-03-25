# Quasi-Dense Similarity Learning for Multiple Object Tracking


This is the updated official implementation of the paper [Quasi-Dense Similarity Learning for Multiple Object Tracking](https://arxiv.org/abs/2006.06664).

## Model Zoo

##### TODO update
We provide pre-trained models for all main experiments:

### BDD100K test set

| mMOTA | mIDF1  | ID Sw. |
|-------|--------|--------|
| 35.5  | 52.3   |  10790 |

### MOT

| Dataset | MOTA | IDF1  | ID Sw. | MT | ML |
|-------|--------|--------| ----| ---| ---|
| MOT16 | 69.8 | 67.1 | 1097 | 316 | 150 |
| MOT17 | 68.7 | 66.3 | 3378 | 957 | 516 |

### Waymo validation set

| Category   | MOTA | IDF1 | ID Sw. |
|------------|------|------|--------|
| Vehicle    | 55.6 | 66.2 | 24309  | 
| Pedestrian | 50.3 | 58.4 | 6347   |
| Cyclist    | 26.2 | 45.7 | 56     | 
| All        | 44.0 | 56.8 | 30712  | 

### TAO

| Split   | AP50 | AP75 | AP | 
|---------|------|------|----|
| val     | 16.1 | 5.0  | 7.0|
| test    | 12.4 | 4.5  | 5.2|

## Training

To reproduce our training, please execute the following:

```
python run.py train --experiment bdd100k --backbone R50FPN --schedule 1x
```