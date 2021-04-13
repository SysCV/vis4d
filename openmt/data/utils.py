"""io utils."""
import sys
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


def im_decode(im_bytes: bytes) -> np.ndarray:
    """Decode to image (numpy array) from bytes."""
    pil_img = Image.open(BytesIO(bytearray(im_bytes)))
    return np.array(pil_img)


def str_decode(str_bytes: bytes, encoding: Optional[str] = None) -> str:
    """Decode to string from bytes."""
    if encoding is None:
        encoding = sys.getdefaultencoding()
    return str_bytes.decode(encoding)
