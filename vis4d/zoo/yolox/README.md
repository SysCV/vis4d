# YOLOX

[YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

Authors: [Zheng Ge](https://joker316701882.github.io/), [Songtao Liu](https://scholar.google.com/citations?user=xY9qK1QAAAAJ), [Feng Wang](https://scholar.google.com/citations?user=ob2gp1QAAAAJ), [Zeming Li](https://www.zemingli.com/), [Jian Sun](http://www.jiansun.org/)

<details>
<summary>Abstract</summary>
In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported. Source code is at [this https URL](https://github.com/Megvii-BaseDetection/YOLOX).
</details>

### Results

| Backbone | Model | Lr schd | Box AP-val | Scores-val |                        Config                         |  Weights  |   Preds   |   Visuals   |
| :------: | :-----: | :------: | :--------: | :--------: | :---------------------------------------------------: | :-------: | :-------: | :---------: |
| CSP-Darknet | Tiny |   300e   |  32.1  | [scores]() | [config](./yolox_tiny_300e_coco.py) | [model]() | [preds]() | [visuals]() |
| CSP-Darknet | S |   300e   |  40.0  | [scores]() | [config](./yolox_s_300e_coco.py) | [model]() | [preds]() | [visuals]() |

## Citation

```bibtex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
