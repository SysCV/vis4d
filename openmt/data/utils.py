"""io utils."""
import cv2
import numpy as np


def im_decode(
    im_bytes: bytes, read_format: int = cv2.IMREAD_COLOR
) -> np.ndarray:
    """Decode image from bytes into numpy array."""
    img_np = np.frombuffer(im_bytes, np.uint8)
    return cv2.imdecode(img_np, read_format)
