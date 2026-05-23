#!/usr/bin/env python3
"""
Breast Cancer Classification - Evaluation Pipeline

Evaluates a trained model on test data and generates:
- Confusion matrix, ROC-AUC curves
- Precision, Recall, F1-Score, Sensitivity, Specificity
- Grad-CAM visualizations
- Classification report
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.model_factory import create_model
from src.models.ensemble import create_ensemble
from src.datasets.dataloader import DataLoaderFactory
from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.visualizer import Visualizer
from src.evaluation.gradcam import GradCAMView
from src.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate Breast Cancer Classification Model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default=None,
                        choices=['efficientnet_b3', 'efficientnet_b4', 'densenet121', 'resnet50'])
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generate Grad-CAM heatmaps')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of Grad-CAM samples per class')
    args = parser.parse_args()

    config = Config(args.config)
    if args.model:
        config.cfg.model.name = args.model
    if args.ensemble:
        config.cfg.model.use_ensemble = True

    logger = Logger(log_dir=os.path.join(config.output.log_dir, 'evaluate'))
    device = config.device
    logger.info(f'Device: {device}')
    logger.info(f'Loading checkpoint: {args.checkpoint}')

    if config.model.use_ensemble:
        model = create_ensemble(config)
    else:
        model = create_model(config)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info('Model loaded successfully')

    loader_factory = DataLoaderFactory(config)
    _, _, test_loader = loader_factory.get_dataloaders()
    logger.info(f'Test samples: {len(test_loader.dataset)}')

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.softmax(model(images), dim=1)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    all_preds = all_outputs.argmax(axis=1)

    metrics_calc = ClassificationMetrics(num_classes=config.model.num_classes)
    report = metrics_calc.compute(all_labels, all_preds, all_outputs)

    logger.info('=' * 50)
    logger.info('EVALUATION RESULTS')
    logger.info('=' * 50)
    logger.info(f'Accuracy:     {report["accuracy"]:.4f}')
    logger.info(f'Precision:    {report["precision"]:.4f}')
    logger.info(f'Recall:       {report["recall"]:.4f}')
    logger.info(f'F1-Score:     {report["f1_score"]:.4f}')
    logger.info(f'Sensitivity:  {report["sensitivity"]:.4f}')
    logger.info(f'Specificity:  {report["specificity"]:.4f}')
    logger.info(f'AUC-ROC:      {report["auc_roc"]:.4f}')
    logger.info(f'Kappa:        {report["kappa"]:.4f}')
    logger.info(f'MCC:          {report["mcc"]:.4f}')
    logger.info('=' * 50)
    logger.info(f'\n{report["classification_report"]}')

    viz = Visualizer(config.output.plot_dir)
    viz.plot_confusion_matrix(
        np.array(report['confusion_matrix']),
        class_names=['benign', 'malignant'],
        filename='test_confusion_matrix.png'
    )
    viz.plot_roc_curve(
        all_labels, all_outputs,
        class_names=['benign', 'malignant'],
        filename='test_roc_curve.png'
    )
    viz.plot_metrics_comparison(report, filename='test_metrics.png')
    viz.save_metrics_report(report, filename='test_metrics_report.txt')

    logger.info(f'Plots saved to {config.output.plot_dir}')

    if args.gradcam:
        logger.info('Generating Grad-CAM visualizations...')
        gradcam_viewer = GradCAMView(
            model, target_layers=[],
            device=device
        )

        loader_factory = DataLoaderFactory(config)
        _, _, test_loader = loader_factory.get_dataloaders()
        dataset = test_loader.dataset

        from src.preprocessing.augmentations import pil_to_numpy
        from PIL import Image

        class_samples = {0: [], 1: []}
        for idx in range(len(dataset)):
            label = dataset.labels[idx]
            if len(class_samples[label]) < args.num_samples:
                class_samples[label].append(idx)

        for class_idx, sample_indices in class_samples.items():
            class_name = ['benign', 'malignant'][class_idx]
            for i, idx in enumerate(sample_indices):
                img_path = dataset.file_paths[idx]
                img = Image.open(img_path).convert('RGB')
                img_np = pil_to_numpy(img)

                from src.preprocessing.augmentations import get_val_transforms
                transforms = get_val_transforms(config)
                img_tensor = transforms(image=img_np)['image'].unsqueeze(0).to(device)

                heatmaps = gradcam_viewer.generate_heatmap(img_tensor, class_idx=class_idx)

                for layer_name, heatmap in heatmaps.items():
                    save_path = os.path.join(
                        config.output.gradcam_dir,
                        f'{class_name}_{i}_{layer_name}.png'
                    )
                    gradcam_viewer.save_heatmap(img_np, heatmap, save_path)
                    logger.info(f'Grad-CAM saved: {save_path}')

    logger.info('Evaluation complete!')


if __name__ == '__main__':
    main()
