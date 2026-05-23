import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
from typing import List, Optional, Dict, Tuple
import numpy as np

from ..config import Config
from .breakhis_dataset import BreaKHisDataset
from .inbreast_dataset import INbreastDataset
from .mias_dataset import MIASDataset
from ..preprocessing.augmentations import get_train_transforms, get_val_transforms
from ..preprocessing.mixup_cutmix import MixUpCutMixCollator


class DataLoaderFactory:
    def __init__(self, config: Config):
        self.config = config

    def create_datasets(self, dataset_names: Optional[List[str]] = None):
        if dataset_names is None:
            dataset_names = self.config.get_active_datasets()

        train_transforms = get_train_transforms(self.config)
        val_transforms = get_val_transforms(self.config)

        all_train_paths, all_train_labels = [], []
        all_val_paths, all_val_labels = [], []

        for name in dataset_names:
            if name == 'breakhis':
                ds = BreaKHisDataset(
                    root=self.config.data.breakhis.path,
                    magnifications=self.config.data.breakhis.get('magnification', None)
                )
            elif name == 'inbreast':
                ds = INbreastDataset(root=self.config.data.inbreast.path)
            elif name == 'mias':
                ds = MIASDataset(root=self.config.data.mias.path)
            else:
                continue

            all_train_paths.extend(ds.file_paths)
            all_train_labels.extend(ds.labels)

        return all_train_paths, all_train_labels

    def get_dataloaders(
        self, dataset_names: Optional[List[str]] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        paths, labels = self.create_datasets(dataset_names)
        if len(paths) == 0:
            raise ValueError(f"No samples found for datasets: {dataset_names}. Check your Drive paths.")
        labels_arr = np.array(labels)
        paths_arr = np.array(paths)

        indices = np.arange(len(paths_arr))
        np.random.seed(self.config.training.seed)
        np.random.shuffle(indices)

        n_total = len(indices)
        n_test = int(n_total * self.config.training.test_split)
        n_val = int(n_total * self.config.training.val_split)
        n_train = n_total - n_test - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_paths = paths_arr[train_idx].tolist()
        train_labels = labels_arr[train_idx].tolist()
        val_paths = paths_arr[val_idx].tolist()
        val_labels = labels_arr[val_idx].tolist()
        test_paths = paths_arr[test_idx].tolist()
        test_labels = labels_arr[test_idx].tolist()

        train_transforms = get_train_transforms(self.config)
        val_transforms = get_val_transforms(self.config)

        from .base_dataset import BaseDataset
        train_ds = BaseDataset(train_paths, train_labels, transform=train_transforms)
        val_ds = BaseDataset(val_paths, val_labels, transform=val_transforms)
        test_ds = BaseDataset(test_paths, test_labels, transform=val_transforms)

        sampler = self._create_sampler(train_labels)

        collator = MixUpCutMixCollator(
            mixup_alpha=self.config.augmentation.mixup_alpha,
            cutmix_alpha=self.config.augmentation.cutmix_alpha,
            prob=self.config.augmentation.mixup_prob,
            num_classes=self.config.model.num_classes
        )

        use_pin = torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            num_workers=self.config.training.num_workers,
            pin_memory=use_pin,
            drop_last=True,
            collate_fn=collator
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=use_pin,
            drop_last=False
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=use_pin,
            drop_last=False
        )

        return train_loader, val_loader, test_loader

    def _create_sampler(self, labels):
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        weights = weights / weights.sum()
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True
        )
        return sampler
