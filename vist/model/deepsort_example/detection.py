"""Class for holding one bounding box detection in a single image."""
import numpy as np
import numpy.typing as npt


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

    def __init__(
        self,
        tlwh: npt.NDArray[np.complex64],
        confidence: float,
        class_id: int,
        feature: npt.NDArray[np.complex64],
    ) -> None:
        """Init."""
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = confidence
        self.class_id = class_id
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self) -> npt.NDArray[np.complex64]:
        """Convert bounding box to format `(min x, min y, max x, max y)`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> npt.NDArray[np.complex64]:
        # pylint: disable=line-too-long
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`.

        aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret
