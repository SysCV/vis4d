# BDD100K Model Zoo

We provide various models trained using Vis4D on the [BDD100K dataset](https://www.vis.xyz/bdd100k/).

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

| Base Network | Iters | Input | mIoU-val | Scores-val | mIoU-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN  |  40K  | 512 \* 1024 |  59.2   | [scores]() |  | [scores]() | [config](./semantic_fpn/semantic_fpn_r50_40k_bdd100k.py) | [model]() | [preds]() | [visuals]() |
