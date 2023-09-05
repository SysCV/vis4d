"""ViT for image classification configs."""

from . import vit_small_imagenet, vit_tiny_imagenet

AVAILABLE_MODELS = {
    "vit_small_imagenet": vit_small_imagenet,
    "vit_tiny_imagenet": vit_tiny_imagenet,
}
