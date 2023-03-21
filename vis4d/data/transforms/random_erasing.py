"""Random erasing data augmentation."""
import numpy as np

from vis4d.common.typing import NDArrayNumber
from vis4d.data.const import CommonKeys as K

from .base import Transform


@Transform(in_keys=K.images, out_keys=K.images)
class RandomErasing:
    """Randomly erase a rectangular region in an image tensor."""

    def __init__(
        self,
        probability: float = 0.5,
        min_area: float = 0.02,
        max_area: float = 0.4,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 1 / 0.3,
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        num_attempt: int = 10,
    ):
        """Creates an instance of RandomErasing.

        Recommended to use this transform after normalization. The erased
        region will be filled with the mean value. See
        `https://arxiv.org/abs/1708.04896`.

        Args:
            probability (float, optional): Probability of applying the
                transform. Defaults to 0.5.
            min_area (float, optional): Minimum area of the erased region.
                Defaults to 0.02.
            max_area (float, optional): Maximum area of the erased region.
                Defaults to 0.4.
            min_aspect_ratio (float, optional): Minimum aspect ratio of the
                erased region. Defaults to 0.3.
            max_aspect_ratio (float, optional): Maximum aspect ratio of the
                erased region. Defaults to 1 / 0.3.
            mean (tuple[float, float, float], optional): Mean of the dataset.
                Defaults to (0.0, 0.0, 0.0).
            num_attempt (int, optional): Number of attempts to find a valid
                erased region. Defaults to 10.

        Returns:
            Callable: A function that takes a tensor of shape [N, C, H, W] and
                returns a tensor of the same shape.
        """
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.mean = mean
        self.num_attempt = num_attempt

    def __call__(self, images: NDArrayNumber) -> NDArrayNumber:
        """Execute the transform."""
        if np.random.rand() > self.probability:
            return images

        fill = np.array(self.mean).reshape((3, 1, 1))
        for i in range(images.shape[0]):
            image = images[i]
            h, w = image.shape[1:3]
            area = h * w

            for _ in range(self.num_attempt):
                target_area = (
                    np.random.uniform(self.min_area, self.max_area) * area
                )
                aspect_ratio = np.random.uniform(
                    self.min_aspect_ratio, self.max_aspect_ratio
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
