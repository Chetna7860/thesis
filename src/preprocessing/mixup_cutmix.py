import torch
import numpy as np
from typing import Optional, List, Tuple


def rand_bbox(size: Tuple[int, int, int, int], lam: float):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixUpCutMixCollator:
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        num_classes: int = 2
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        labels = torch.tensor(labels, dtype=torch.long)

        if np.random.random() < self.prob:
            return images, labels

        if np.random.random() < 0.5:
            return self._mixup(images, labels)
        else:
            return self._cutmix(images, labels)

    def _mixup(self, images: torch.Tensor, labels: torch.Tensor):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]

        return mixed_images, (labels_a, labels_b, lam)

    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels_a = labels
        labels_b = labels[index]

        return images, (labels_a, labels_b, lam)


class MixUpCutMixLoss(torch.nn.Module):
    def __init__(self, criterion: torch.nn.Module):
        super().__init__()
        self.criterion = criterion

    def forward(self, preds, target):
        if isinstance(target, tuple):
            labels_a, labels_b, lam = target
            return lam * self.criterion(preds, labels_a) + (1 - lam) * self.criterion(preds, labels_b)
        return self.criterion(preds, target)
