import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import gc
from typing import Optional, Tuple, Dict, Any, Callable

from ..config import Config
from ..utils.early_stopping import EarlyStopping
from ..utils.logger import Logger
from ..evaluation.metrics import ClassificationMetrics
from ..preprocessing.mixup_cutmix import MixUpCutMixLoss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        run_name: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = config.device
        self.run_name = run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.training.label_smoothing
        )
        self.mixup_criterion = MixUpCutMixLoss(self.criterion)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        total_steps = config.training.epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.training.min_lr
        )

        self.scaler = GradScaler('cuda') if (config.training.mixed_precision and torch.cuda.is_available()) else None
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stop_patience,
            verbose=True
        )
        self.logger = Logger(
            log_dir=os.path.join(config.output.log_dir, self.run_name),
            tensorboard=True
        )
        self.writer = SummaryWriter(log_dir=os.path.join(config.output.log_dir, self.run_name))

        self.best_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)

            if isinstance(labels, tuple):
                labels_a, labels_b, lam = labels
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                labels = (labels_a, labels_b, lam)
                criterion = self.mixup_criterion
            else:
                labels = labels.to(self.device)
                criterion = self.criterion

            self.optimizer.zero_grad()

            if self.scaler and torch.cuda.is_available():
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )
                self.optimizer.step()

            total_loss += loss.item()

            if isinstance(labels, tuple):
                preds = outputs.argmax(dim=1)
                correct += (preds == labels_a).float().sum().item()
                correct += (preds == labels_b).float().sum().item()
                total += 2 * labels_a.size(0)
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / max(total, 1):.4f}'
            })

            if batch_idx % 50 == 0:
                gc.collect()

        gc.collect()
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean()

        return avg_loss, accuracy, {
            'preds': all_preds,
            'labels': all_labels
        }

    def fit(self, train_loader, val_loader, epochs: Optional[int] = None):
        epochs = epochs or self.config.training.epochs

        for epoch in range(epochs):
            self.logger.info(f'Epoch {epoch + 1}/{epochs}')

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_info = self.validate(val_loader)

            gc.collect()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)

            self.logger.info(
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}'
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_acc, val_loss, is_best=True)
                self.logger.info(f'New best model saved! Val Acc: {val_acc:.4f}')

            self._save_checkpoint(epoch, val_acc, val_loss, is_best=False)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                self.logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break

        self.logger.info(
            f'Training completed. Best epoch: {self.best_epoch + 1}, '
            f'Best val acc: {self.best_val_acc:.4f}'
        )
        self.writer.close()

        return self.best_val_acc

    def _save_checkpoint(self, epoch: int, val_acc: float, val_loss: float, is_best: bool = False):
        suffix = 'best' if is_best else f'epoch_{epoch + 1}'
        path = os.path.join(
            self.config.output.checkpoint_dir,
            self.run_name,
            f'model_{suffix}.pth'
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': self.config.cfg,
        }, path)
