"""Class for holding one bounding box detection in a single image."""
import numpy as np


class Detection:
    """This class represents one bounding box detection in a single image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, class_id, feature):
        """Init."""
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = confidence
        self.class_id = class_id
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        # pylint: disable=line-too-long
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`.

        aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret
