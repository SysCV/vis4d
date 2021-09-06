"""Class for holding one bounding box detection in a single image."""
import torch


class Detection:
    """This class represents one bounding box detection in a single image.

    Attributes
    ----------
    tlwh : torch.tensor
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : float
        Detector confidence score.
    feature : torch.tensor
        A feature vector that describes the object contained in this image.
    """

    def __init__(
        self,
        tlwh: torch.tensor,
        confidence: float,
        class_id: int,
        feature: torch.tensor,
    ):
        """Init."""
        self.tlwh = tlwh.clone().detach()
        self.confidence = confidence
        self.class_id = class_id
        self.feature = feature.clone().detach()

    def to_tlbr(self) -> torch.tensor:
        """Convert bounding box to format `(min x, min y, max x, max y)`."""
        ret = self.tlwh.clone().detach()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> torch.tensor:
        # pylint: disable=line-too-long
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`.

        aspect ratio is `width / height`.
        """
        ret = self.tlwh.clone().detach()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret
