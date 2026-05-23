import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from typing import Callable
from ..config import Config


def get_train_transforms(config: Config) -> Callable:
    img_size = config.augmentation.img_size
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=config.augmentation.rotation, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=config.augmentation.brightness,
            contrast_limit=config.augmentation.contrast,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=int(config.augmentation.hue * 180),
            sat_shift_limit=int(config.augmentation.saturation * 255),
            val_shift_limit=0,
            p=0.3
        ),
        A.RandomGamma(p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), per_channel=True, p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.125),
            hole_width_range=(0.05, 0.125),
            fill=0, p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={})


def get_val_transforms(config: Config) -> Callable:
    img_size = config.augmentation.img_size
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_transforms(config: Config) -> Callable:
    return get_val_transforms(config)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.uint8)
