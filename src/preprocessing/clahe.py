import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, grid_size: int = 8):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        enhanced = apply_clahe(img_np, self.clip_limit, self.grid_size)
        return Image.fromarray(enhanced)
