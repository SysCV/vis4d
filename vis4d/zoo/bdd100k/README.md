# BDD100K Model Zoo

We provide various models trained using Vis4D on the [BDD100K dataset](https://www.vis.xyz/bdd100k/).

## Object Detection

The object detection task involves localization (predicting a bounding box for each object) and classification (predicting the object category).

The BDD100K dataset contains bounding box annotations for 100K images (70K/10K/20K for train/val/test). Each annotation contains bounding box labels for 10 object classes. For details about downloading the data and the annotation format for this task, see the [official documentation](https://doc.bdd100k.com/download.html).

### Faster R-CNN

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) [NeurIPS 2015]

Authors: [Shaoqing Ren](https://www.shaoqingren.com/), [Kaiming He](http://kaiminghe.com/), [Ross Girshick](https://www.rossgirshick.info/), [Jian Sun](http://www.jiansun.org/)

<details>
<summary>Abstract</summary>
State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test |                        Config                         |  Weights  |   Preds   |   Visuals   |
| :------: | :-----: | :------: | :--------: | :--------: | :---------: | :---------: | :---------------------------------------------------: | :-------: | :-------: | :---------: |
| R-50-FPN |   1x    |          |    31.2    | [scores]() |             | [scores]()  | [config](./faster_rcnn/faster_rcnn_r50_1x_bdd100k.py) | [model]() | [preds]() | [visuals]() |
| R-50-FPN |   3x    |    ✓     |    32.4    | [scores]() |             | [scores]()  | [config](./faster_rcnn/faster_rcnn_r50_3x_bdd100k.py) | [model]() | [preds]() | [visuals]() |

---

## Instance Segmentation

The instance segmentation task involves detecting and segmenting each distinct object of interest in the scene.

The BDD100K dataset contains object segmentation annotations for 10K images (7K/1K/2K for train/val/test). Each annotation contains labels for 8 object classes. For details about downloading the data and the annotation format for this task, see the [official documentation](https://doc.bdd100k.com/download.html).

### Mask R-CNN

[Mask R-CNN](https://arxiv.org/abs/1703.06870) [ICCV 2017]

Authors: [Kaiming He](http://kaiminghe.com/), [Georgia Gkioxari](https://gkioxari.github.io/), [Piotr Dollár](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

<details>
<summary>Abstract</summary>
We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: [this https URL](https://github.com/facebookresearch/detectron2).
</details>

#### Results

| Backbone | Lr schd | MS-train | Mask AP-val | Box AP-val | Scores-val | Mask AP-test | Box AP-test | Scores-test |                      Config                       |  Weights  |   Preds   |   Visuals   |
| :------: | :-----: | :------: | :---------: | :--------: | :--------: | :----------: | :---------: | :---------: | :-----------------------------------------------: | :-------: | :-------: | :---------: |
| R-50-FPN |   1x    |          |    17.0     |    23.4    | [scores]() |              |             | [scores]()  | [config](./mask_rcnn/mask_rcnn_r50_1x_bdd100k.py) | [model]() | [preds]() | [visuals]() |
| R-50-FPN |   3x    |    ✓     |    20.2     |    26.8    | [scores]() |              |             | [scores]()  | [config](./mask_rcnn/mask_rcnn_r50_3x_bdd100k.py) | [model]() | [preds]() | [visuals]() |
| R-50-FPN |   5x    |    ✓     |    20.2     |    25.7    | [scores]() |              |             | [scores]()  | [config](./mask_rcnn/mask_rcnn_r50_5x_bdd100k.py) | [model]() | [preds]() | [visuals]() |

---

## Semantic Segmentation

The semantic segmentation task involves predicting a segmentation mask for each image indicating a class label for every pixel.

The BDD100K dataset contains fine-grained semantic segmentation annotations for 10K images (7K/1K/2K for train/val/test). Each annotation is a segmentation mask containing labels for 19 diverse object classes. For details about downloading the data and the annotation format for this task, see the [official documentation](https://doc.bdd100k.com/download.html).

### Semantic FPN

[Panoptic Feature Pyramid Networks](https://arxiv.org/abs/1901.02446) [CVPR 2019]

Authors: [Alexander Kirillov](https://alexander-kirillov.github.io/), [Ross Girshick](https://www.rossgirshick.info/), [Kaiming He](http://kaiminghe.com/), [Piotr Dollár](https://pdollar.github.io/)

<details>
<summary>Abstract</summary>
The recently introduced panoptic segmentation task has renewed our community's interest in unifying the tasks of instance segmentation (for thing classes) and semantic segmentation (for stuff classes). However, current state-of-the-art methods for this joint task use separate and dissimilar networks for instance and semantic segmentation, without performing any shared computation. In this work, we aim to unify these methods at the architectural level, designing a single network for both tasks. Our approach is to endow Mask R-CNN, a popular instance segmentation method, with a semantic segmentation branch using a shared Feature Pyramid Network (FPN) backbone. Surprisingly, this simple baseline not only remains effective for instance segmentation, but also yields a lightweight, top-performing method for semantic segmentation. In this work, we perform a detailed study of this minimally extended version of Mask R-CNN with FPN, which we refer to as Panoptic FPN, and show it is a robust and accurate baseline for both tasks. Given its effectiveness and conceptual simplicity, we hope our method can serve as a strong baseline and aid future research in panoptic segmentation.
</details>

#### Results

| Base Network | Iters |    Input    | mIoU-val | Scores-val | mIoU-test | Scores-test |                          Config                          |  Weights  |   Preds   |   Visuals   |
| :----------: | :---: | :---------: | :------: | :--------: | :-------: | :---------: | :------------------------------------------------------: | :-------: | :-------: | :---------: |
|   R-50-FPN   |  40K  | 512 \* 1024 |   59.2   | [scores]() |           | [scores]()  | [config](./semantic_fpn/semantic_fpn_r50_40k_bdd100k.py) | [model]() | [preds]() | [visuals]() |

---

## Multiple Object Tracking

The multiple object tracking (MOT) task involves detecting and tracking objects of interest throughout each video sequence.

The BDD100K dataset contains MOT annotations for 2K videos (1.4K/200/400 for train/val/test) with 8 categories. Each video is approximately 40 seconds and annotated at 5 fps, resulting in around 200 frames per video. For details about downloading the data and the annotation format for this task, see the [official documentation](https://doc.bdd100k.com/download.html).

### QDTrack

[Quasi-Dense Similarity Learning for Multiple Object Tracking](https://arxiv.org/abs/2006.06664) [CVPR 2021 Oral]

Authors: [Jiangmiao Pang](https://scholar.google.com/citations?user=ssSfKpAAAAAJ), [Linlu Qiu](https://linlu-qiu.github.io/), [Xia Li](https://xialipku.github.io/), [Haofeng Chen](https://www.haofeng.io/), Qi Li, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Fisher Yu](https://www.yf.io/)

<details>
<summary>Abstract</summary>
Similarity learning has been recognized as a crucial step for object tracking. However, existing multiple object tracking methods only use sparse ground truth matching as the training objective, while ignoring the majority of the informative regions on the images. In this paper, we present Quasi-Dense Similarity Learning, which densely samples hundreds of region proposals on a pair of images for contrastive learning. We can naturally combine this similarity learning with existing detection methods to build Quasi-Dense Tracking (QDTrack) without turning to displacement regression or motion priors. We also find that the resulting distinctive feature space admits a simple nearest neighbor search at the inference time. Despite its simplicity, QDTrack outperforms all existing methods on MOT, BDD100K, Waymo, and TAO tracking benchmarks. It achieves 68.7 MOTA at 20.3 FPS on MOT17 without using external training data. Compared to methods with similar detectors, it boosts almost 10 points of MOTA and significantly decreases the number of ID switches on BDD100K and Waymo datasets.
</details>

#### Results

| Detector  | Base Network | mMOTA-val | mIDF1-val | ID Sw.-val | Scores-val | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Faster R-CNN | R-50-FPN |  |  |  | [scores]() | [config]() | [model]() | [preds]() | [visuals]() |
| YOLOX-x | CSPNet |  |  |  | [scores]() | [config]() | [model]() | [preds]() | [visuals]() |
