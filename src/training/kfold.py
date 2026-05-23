import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Tuple, Dict, Any
import os
import copy
import time

from ..config import Config
from ..models.model_factory import create_model
from ..training.trainer import Trainer
from ..datasets.base_dataset import BaseDataset
from ..datasets.breakhis_dataset import BreaKHisDataset
from ..datasets.inbreast_dataset import INbreastDataset
from ..datasets.mias_dataset import MIASDataset
from ..preprocessing.augmentations import get_train_transforms, get_val_transforms
from ..utils.logger import Logger


class KFoldTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(log_dir=os.path.join(config.output.log_dir, 'kfold'))
        self.device = config.device

    def run(self, n_splits: int = 5):
        paths, labels = self._load_all_data()
        labels = np.array(labels)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.training.seed)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
            self.logger.info(f'=== Fold {fold + 1}/{n_splits} ===')

            train_paths = [paths[i] for i in train_idx]
            train_labels = labels[train_idx].tolist()
            val_paths = [paths[i] for i in val_idx]
            val_labels = labels[val_idx].tolist()

            train_transforms = get_train_transforms(self.config)
            val_transforms = get_val_transforms(self.config)

            train_ds = BaseDataset(train_paths, train_labels, transform=train_transforms)
            val_ds = BaseDataset(val_paths, val_labels.tolist(), transform=val_transforms)

            sampler = self._create_sampler(train_labels)

            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.training.batch_size,
                sampler=sampler,
                num_workers=self.config.training.num_workers,
                pin_memory=True,
                drop_last=True
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=self.config.training.batch_size * 2,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                pin_memory=True,
                drop_last=False
            )

            model = create_model(self.config)
            trainer = Trainer(
                model,
                self.config,
                run_name=f'kfold_fold_{fold + 1}'
            )
            best_acc = trainer.fit(train_loader, val_loader)
            fold_results.append(best_acc)

            self.logger.info(f'Fold {fold + 1} best val acc: {best_acc:.4f}')

        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        self.logger.info(f'K-Fold Cross Validation Results:')
        self.logger.info(f'Individual folds: {[f"{r:.4f}" for r in fold_results]}')
        self.logger.info(f'Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}')

        return fold_results, mean_acc, std_acc

    def _load_all_data(self) -> Tuple[List[str], List[int]]:
        all_paths, all_labels = [], []
        dataset_names = self.config.get_active_datasets()

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
            all_paths.extend(ds.file_paths)
            all_labels.extend(ds.labels)

        return all_paths, all_labels

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
