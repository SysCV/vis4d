"""io utils."""
import cv2
import numpy as np


def im_decode(im_bytes, format=cv2.IMREAD_COLOR):
    img_np = np.frombuffer(im_bytes, np.uint8)
    return cv2.imdecode(img_np, format)
