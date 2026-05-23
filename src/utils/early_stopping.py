import torch
import numpy as np
from typing import Optional, Callable


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        verbose: bool = False,
        delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.inf

    def __call__(self, val_loss: float):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_loss = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_loss = val_loss
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
