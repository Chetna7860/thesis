import os
import glob
from typing import List, Tuple, Optional, Callable
from .base_dataset import BaseDataset


class BreaKHisDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        magnifications: Optional[List[int]] = None
    ):
        file_paths, labels = self._collect_samples(root, magnifications)
        class_names = ['benign', 'malignant']
        super().__init__(file_paths, labels, transform, class_names)

    def _collect_samples(
        self, root: str, magnifications: Optional[List[int]]
    ) -> Tuple[List[str], List[int]]:
        file_paths = []
        labels = []

        if not os.path.exists(root):
            raise FileNotFoundError(f"BreaKHis root not found: {root}")

        label_map = {'benign': 0, 'malignant': 1}
        for class_name, label in label_map.items():
            sob_dir = os.path.join(root, class_name, 'SOB')
            if not os.path.exists(sob_dir):
                continue

            all_images = glob.glob(os.path.join(sob_dir, '**', '*.png'), recursive=True)

            if magnifications:
                for mag in magnifications:
                    mag_tag = f'-{mag}-'
                    mag_filtered = [p for p in all_images if mag_tag in os.path.basename(p)]
                    for p in mag_filtered:
                        file_paths.append(p)
                        labels.append(label)
            else:
                for p in all_images:
                    file_paths.append(p)
                    labels.append(label)

        return file_paths, labels
