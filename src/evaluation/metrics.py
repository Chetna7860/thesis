import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef
)
from typing import Dict, Any, Tuple, List, Optional


class ClassificationMetrics:
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        metrics = {}

        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        if self.num_classes == 2:
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
            metrics['specificity'] = self._specificity(y_true, y_pred)
            metrics['sensitivity'] = metrics['recall']
            metrics['auc_roc'] = float(
                roc_auc_score(y_true, y_prob[:, 1]) if y_prob is not None else 0.0
            )
        else:
            metrics['precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['auc_roc'] = float(
                roc_auc_score(y_true, y_prob, multi_class='ovr') if y_prob is not None else 0.0
            )

        metrics['kappa'] = float(cohen_kappa_score(y_true, y_pred))
        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, labels=range(self.num_classes), zero_division=0
        )

        return metrics

    def _specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return 0.0
