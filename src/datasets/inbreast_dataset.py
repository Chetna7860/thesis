import os
import pandas as pd
import numpy as np
import pydicom
import tarfile
import tempfile
import shutil
from typing import List, Tuple, Optional, Callable
from .base_dataset import BaseDataset


class INbreastDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extract: bool = True
    ):
        self._temp_dir = None
        if extract:
            root = self._extract_tgz(root)
        file_paths, labels = self._collect_samples(root)
        class_names = ['normal', 'abnormal']
        super().__init__(file_paths, labels, transform, class_names)
        if extract and self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _extract_tgz(self, root: str) -> str:
        tgz_path = os.path.join(root, 'inbreast.tgz')
        if not os.path.exists(tgz_path):
            return root

        xls_path = os.path.join(root, 'INbreast.xls')
        img_dir = os.path.join(root, 'ALL-IMGS')

        if os.path.exists(xls_path) and os.path.exists(img_dir):
            return root

        self._temp_dir = tempfile.mkdtemp()
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=self._temp_dir)
        return self._temp_dir

    def _collect_samples(self, root: str) -> Tuple[List[str], List[int]]:
        file_paths = []
        labels = []

        xls_path = os.path.join(root, 'INbreast.xls')
        if os.path.exists(xls_path):
            df = pd.read_excel(xls_path)
            if 'File Name' in df.columns and 'Biopsy' in df.columns:
                label_map = {0: 0, 1: 1}
                for _, row in df.iterrows():
                    fname = str(row['File Name']).strip()
                    biopsy = int(row['Biopsy'])
                    img_path = os.path.join(root, 'ALL-IMGS', fname)
                    if os.path.exists(img_path):
                        file_paths.append(img_path)
                        labels.append(label_map.get(biopsy, 0))
                return file_paths, labels

        img_dir = os.path.join(root, 'ALL-IMGS')
        if os.path.exists(img_dir):
            dcm_files = [f for f in os.listdir(img_dir) if f.endswith('.dcm')]
            for f in dcm_files:
                file_paths.append(os.path.join(img_dir, f))
                labels.append(0)

        return file_paths, labels
