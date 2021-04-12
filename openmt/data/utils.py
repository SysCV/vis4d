"""io utils."""
import sys
from typing import Optional

import cv2
import numpy as np


def im_decode(
    im_bytes: bytes, read_format: int = cv2.IMREAD_COLOR
) -> np.ndarray:
    """Decode to image (numpy array) from bytes."""
    img_np = np.frombuffer(im_bytes, np.uint8)
    return cv2.imdecode(img_np, read_format)


def str_decode(str_bytes: bytes, encoding: Optional[str] = None) -> str:
    """Decode to string from bytes."""
    if encoding is None:
        encoding = sys.getdefaultencoding()
    return str_bytes.decode(encoding)
