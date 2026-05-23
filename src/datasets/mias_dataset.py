import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Callable
from .base_dataset import BaseDataset


class MIASDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        binary: str = 'benign_vs_malignant'
    ):
        file_paths, labels = self._collect_samples(root, binary)
        class_names = ['benign', 'malignant'] if binary == 'benign_vs_malignant' else ['normal', 'abnormal']
        super().__init__(file_paths, labels, transform, class_names)

    def _collect_samples(self, root: str, binary: str) -> Tuple[List[str], List[int]]:
        file_paths = []
        labels = []

        info_path = os.path.join(root, 'Info.txt')
        if not os.path.exists(info_path):
            alt = os.path.join(root, 'all-mias', 'Info.txt')
            if os.path.exists(alt):
                info_path = alt
            else:
                return file_paths, labels

        img_dir = os.path.join(root, 'all-mias')
        if not os.path.exists(img_dir):
            img_dir = root

        pgm_dir = img_dir

        with open(info_path, 'r') as f:
            lines = f.readlines()

        scanner_line = lines[0].strip() if lines else ''
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            refnum = parts[0]
            bg = parts[1]
            cls_type = parts[2]
            severity = parts[3] if len(parts) > 3 else ''

            img_path = os.path.join(pgm_dir, f'{refnum}.pgm')
            if not os.path.exists(img_path):
                continue

            file_paths.append(img_path)

            if binary == 'benign_vs_malignant':
                if cls_type == 'NORM':
                    labels.append(0)
                elif severity == 'B':
                    labels.append(0)
                elif severity == 'M':
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                if cls_type == 'NORM':
                    labels.append(0)
                else:
                    labels.append(1)

        return file_paths, labels
