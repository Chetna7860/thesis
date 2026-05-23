import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from typing import Optional, List, Dict, Any
import pandas as pd


class Visualizer:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.rcParams.update({'font.size': 12})

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = 'Confusion Matrix',
        filename: str = 'confusion_matrix.png'
    ):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = 'ROC Curves',
        filename: str = 'roc_curve.png'
    ):
        plt.figure(figsize=(8, 6))
        n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                label=f'ROC curve (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        else:
            for i in range(n_classes):
                y_true_bin = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                label = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        filename: str = 'training_history.png'
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()

    def plot_metrics_comparison(
        self,
        metrics: Dict[str, Any],
        filename: str = 'metrics_comparison.png'
    ):
        plot_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
        values = []
        labels = []
        for m in plot_metrics:
            if m in metrics:
                values.append(metrics[m])
                labels.append(m.replace('_', ' ').title())

        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(values)))
        bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)

        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold'
            )

        plt.ylim(0, 1.1)
        plt.title('Classification Metrics')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()

    def save_metrics_report(self, metrics: Dict[str, Any], filename: str = 'metrics_report.txt'):
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            f.write('=' * 60 + '\n')
            f.write('Classification Metrics Report\n')
            f.write('=' * 60 + '\n\n')

            skip_keys = {'confusion_matrix', 'classification_report'}
            for key, value in metrics.items():
                if key in skip_keys:
                    continue
                if isinstance(value, (int, float)):
                    f.write(f'{key.replace("_", " ").title():20s}: {value:.4f}\n')

            if 'confusion_matrix' in metrics:
                f.write('\nConfusion Matrix:\n')
                cm = np.array(metrics['confusion_matrix'])
                f.write(f'{cm}\n')

            if 'classification_report' in metrics:
                f.write('\nClassification Report:\n')
                f.write(metrics['classification_report'])

        print(f'Metrics report saved to {path}')
