"""Random erasing data augmentation."""
import numpy as np
import torch

from vis4d.data.const import CommonKeys as K

from .base import Transform


@Transform(
    in_keys=(K.images,),
    out_keys=(K.images,),
)
def random_erasing(
    probability: float = 0.5,
    min_area: float = 0.02,
    max_area: float = 0.4,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 1 / 0.3,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    num_attempt: int = 10,
):
    """Randomly erase a rectangular region in an image tensor.

    Recommended to use this transform after normalization. The erased region
    will be filled with the mean value. See `https://arxiv.org/abs/1708.04896`.

    Args:
        probability (float, optional): Probability of applying the transform.
            Defaults to 0.5.
        min_area (float, optional): Minimum area of the erased region. Defaults
            to 0.02.
        max_area (float, optional): Maximum area of the erased region. Defaults
            to 0.4.
        min_aspect_ratio (float, optional): Minimum aspect ratio of the erased
            region. Defaults to 0.3.
        max_aspect_ratio (float, optional): Maximum aspect ratio of the erased
            region. Defaults to 1 / 0.3.
        mean (tuple[float, float, float], optional): Mean of the dataset.
            Defaults to (0.0, 0.0, 0.0).
        num_attempt (int, optional): Number of attempts to find a valid erased
            region. Defaults to 10.

    Returns:
        Callable: A function that takes a tensor of shape [N, C, H, W] and
            returns a tensor of the same shape.
    """

    def _random_erasing(images: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > probability:
            return images

        device = images.device
        fill = torch.tensor(mean, device=device).view(3, 1, 1)
        for i in range(images.size(0)):
            image = images[i]
            h, w = image.size(1), image.size(2)
            area = h * w

            for _ in range(num_attempt):
                target_area = np.random.uniform(min_area, max_area) * area
                aspect_ratio = np.random.uniform(
                    min_aspect_ratio, max_aspect_ratio
                )
                h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
                w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
                if w_erase < w and h_erase < h:
                    x_erase = np.random.randint(0, w - w_erase)
                    y_erase = np.random.randint(0, h - h_erase)
                    image[
                        :,
                        y_erase : y_erase + h_erase,
                        x_erase : x_erase + w_erase,
                    ] = fill
                    break
        return images

    return _random_erasing
