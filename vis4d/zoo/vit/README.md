# Vision Transformer



<!-- [ALGORITHM] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142579081-b5718032-6581-472b-8037-ea66aaa9e278.png" width="70%"/>
</div>

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) [ICLR 2021]

Authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

<details>

<summary>Abstract</summary>

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
</br>

</details>

### Usage

<!-- [TABS-BEGIN] -->

**Use the model**

```python
from vis4d.model.cls import ViTClassifer

vit_classifer = ViTClassifer(
    variant="vit_small_patch16_224",    # specify the model variant
    embed_dim=192,                      # support to override the variant's args
    num_heads=192,
    drop_path_rate=0.1,
    num_classes=1000,
    weights=None,
)
out = vit_classifer(images)
print(out.probs)
```

**Load pretrained weights**

```python
from vis4d.model.cls import ViTClassifer

vit_classifer = ViTClassifer(
    variant="vit_small_patch16_224",
    # Support Timm's model names: {model_variant}.{pretrained_tag}
    pretrained="timm://vit_small_patch16_224.augreg_in21k_ft_in1k",
)
```


<!-- [TABS-END] -->

### Results

| Model                                           |   Train data   | Params (M) | Top-1 (%) | Top-5 (%) |                    Config                    |                           Download                           |
| :---------------------------------------------- | :----------: | :--------: | :-------: | :-------: | :------------------------------------------: | :----------------------------------------------------------: |
| vit-tiny-patch16  | ImageNet-1k |   11.45   |   72.51   |   90.75   | [config](vit_tiny_imagenet.py)  | [model](https://dl.cv.ethz.ch/vis4d/vit/vit_tiny_patch16_imagenet1k_ccdf98.pth) |
| vit-small-patch16  | ImageNet-1k |   44.15   |   78.31   |   93.53   | [config](vit_small_imagenet.py)  | [model](https://dl.cv.ethz.ch/vis4d/vit/vit_small_patch16_imagenet1k_3773a7.pth) |
| vit-base-patch16\* | ImageNet-1k |   86.57   |   82.37	  |   96.15   | [config](vit_base_imagenet.py)  | [model]() |


*Models with * are converted from the [official repo](https://github.com/google-research/vision_transformer/blob/88a52f8892c80c10de99194990a517b4d80485fd/vit_jax/models.py#L208). The config files of these models are only for inference. We haven't reprodcue the training results.*

## Citation

```bibtex
@inproceedings{
  dosovitskiy2021an,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=YicbFdNTTy}
}
```
