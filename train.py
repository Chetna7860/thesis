#!/usr/bin/env python3
"""
Breast Cancer Classification - Training Pipeline

Supports:
- Multi-dataset training (BreaKHis, INbreast, MIAS)
- Transfer learning (EfficientNet-B3/B4, DenseNet121, ResNet50)
- Ensemble learning
- Stratified K-fold cross-validation
- Mixed precision training
- CLAHE preprocessing, augmentations, MixUp/CutMix
- TensorBoard logging, checkpointing, early stopping
"""

import os
import sys
import argparse
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.model_factory import create_model, ModelFactory
from src.models.ensemble import create_ensemble
from src.datasets.dataloader import DataLoaderFactory
from src.training.trainer import Trainer
from src.training.kfold import KFoldTrainer
from src.utils.logger import Logger


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Classification Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        choices=['cnn', 'efficientnet_b3', 'efficientnet_b4', 'densenet121', 'resnet50', 'mobilenet_v3_small'],
                        help='Model architecture')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of all models')
    parser.add_argument('--kfold', type=int, default=0,
                        help='Number of K-fold splits (0 = standard train/val/test)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        choices=['breakhis', 'inbreast', 'mias'],
                        help='Datasets to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--magnification', type=int, nargs='+', default=None,
                        help='BreaKHis magnification levels (e.g. 40 100 200 400)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    args = parser.parse_args()

    config = Config(args.config)

    if args.model:
        config.cfg.model.name = args.model
    if args.ensemble:
        config.cfg.model.use_ensemble = True
    if args.epochs:
        config.cfg.training.epochs = args.epochs
    if args.batch_size:
        config.cfg.training.batch_size = args.batch_size
    if args.lr:
        config.cfg.training.learning_rate = args.lr
    if args.seed:
        config.cfg.training.seed = args.seed
    if args.magnification:
        config.cfg.data.breakhis.magnification = args.magnification

    if args.datasets:
        for ds_name in ['breakhis', 'inbreast', 'mias']:
            config.cfg.data[ds_name].use = ds_name in args.datasets

    set_seed(config.training.seed)

    logger = Logger(log_dir=os.path.join(config.output.log_dir, 'train'))
    logger.info(f'Device: {config.device}')
    logger.info(f'Model: {config.model.name}')
    logger.info(f'Ensemble: {config.model.use_ensemble}')
    logger.info(f'Datasets: {config.get_active_datasets()}')
    logger.info(f'Config: {config.cfg}')

    if args.kfold > 1:
        logger.info(f'Starting {args.kfold}-fold cross-validation...')
        kfold_trainer = KFoldTrainer(config)
        fold_results, mean_acc, std_acc = kfold_trainer.run(n_splits=args.kfold)
        logger.info(f'K-Fold Complete - Mean: {mean_acc:.4f} ± {std_acc:.4f}')
        return

    if config.model.use_ensemble:
        logger.info('Creating ensemble model...')
        model = create_ensemble(config)
    else:
        model = create_model(config)

    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Checkpoint loaded (epoch {checkpoint.get("epoch", "unknown")})')

    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    loader_factory = DataLoaderFactory(config)
    train_loader, val_loader, test_loader = loader_factory.get_dataloaders()

    logger.info(f'Train samples: {len(train_loader.dataset)}')
    logger.info(f'Val samples: {len(val_loader.dataset)}')
    logger.info(f'Test samples: {len(test_loader.dataset)}')

    trainer = Trainer(model, config)
    best_acc = trainer.fit(train_loader, val_loader)

    logger.info(f'Training complete. Best val accuracy: {best_acc:.4f}')

    logger.info('Evaluating on test set...')
    test_loss, test_acc, test_info = trainer.validate(test_loader)

    from src.evaluation.metrics import ClassificationMetrics
    metrics = ClassificationMetrics(num_classes=config.model.num_classes)

    import numpy as np
    all_outputs = []
    all_test_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            outputs = torch.softmax(model(images), dim=1)
            all_outputs.append(outputs.cpu().numpy())
            all_test_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    all_preds = all_outputs.argmax(axis=1)

    report = metrics.compute(all_test_labels, all_preds, all_outputs)
    logger.info(f'Test Accuracy: {report["accuracy"]:.4f}')
    logger.info(f'Test F1: {report["f1_score"]:.4f}')
    logger.info(f'Test AUC-ROC: {report["auc_roc"]:.4f}')
    logger.info(f'Test Sensitivity: {report["sensitivity"]:.4f}')
    logger.info(f'Test Specificity: {report["specificity"]:.4f}')

    from src.evaluation.visualizer import Visualizer
    viz = Visualizer(config.output.plot_dir)
    viz.plot_confusion_matrix(
        np.array(report['confusion_matrix']),
        class_names=['benign', 'malignant']
    )
    viz.plot_roc_curve(all_test_labels, all_outputs, class_names=['benign', 'malignant'])
    viz.plot_training_history(
        trainer.train_losses, trainer.val_losses,
        trainer.train_accs, trainer.val_accs
    )
    viz.plot_metrics_comparison(report)
    viz.save_metrics_report(report)

    model_path = os.path.join(config.output.model_dir, f'{config.model.name}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.cfg,
        'test_accuracy': report['accuracy'],
        'test_report': report
    }, model_path)
    logger.info(f'Final model saved to {model_path}')


if __name__ == '__main__':
    main()
