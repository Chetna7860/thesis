import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Tuple


class BaseDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or ['benign', 'malignant']

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        label = self.labels[idx]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, label

    def get_class_weights(self):
        from collections import Counter
        import torch
        counts = Counter(self.labels)
        total = len(self.labels)
        weights = [total / (len(counts) * counts[c]) for c in sorted(counts.keys())]
        return torch.tensor(weights, dtype=torch.float)
