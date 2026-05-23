#!/usr/bin/env python3
"""
Breast Cancer Classification - Inference Pipeline

Supports:
- Single image prediction
- Batch image prediction
- Grad-CAM overlay on predictions
- Model ensembling for predictions
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.model_factory import create_model
from src.models.ensemble import create_ensemble
from src.preprocessing.augmentations import get_test_transforms, pil_to_numpy
from src.evaluation.gradcam import GradCAMView


class Predictor:
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.device = config.device

        if config.model.use_ensemble:
            self.model = create_ensemble(config)
        else:
            self.model = create_model(config)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = get_test_transforms(config)
        self.class_names = ['benign', 'malignant']

    def predict(self, image_path: str) -> Tuple[int, str, float]:
        image = Image.open(image_path).convert('RGB')
        img_np = pil_to_numpy(image)
        transformed = self.transform(image=img_np)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = torch.softmax(self.model(input_tensor), dim=1)
            confidence, pred_class = outputs.max(dim=1)

        class_idx = pred_class.item()
        class_name = self.class_names[class_idx]
        confidence_score = confidence.item()

        return class_idx, class_name, confidence_score

    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, str, float]]:
        results = []
        for path in image_paths:
            _, class_name, confidence = self.predict(path)
            results.append((path, class_name, confidence))
        return results

    def predict_with_gradcam(
        self, image_path: str, save_path: Optional[str] = None
    ) -> Tuple[int, str, float, np.ndarray]:
        image = Image.open(image_path).convert('RGB')
        img_np = pil_to_numpy(image)
        transformed = self.transform(image=img_np)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = torch.softmax(self.model(input_tensor), dim=1)
            confidence, pred_class = outputs.max(dim=1)

        class_idx = pred_class.item()
        class_name = self.class_names[class_idx]
        confidence_score = confidence.item()

        gradcam_viewer = GradCAMView(self.model, target_layers=[], device=self.device)
        heatmaps = gradcam_viewer.generate_heatmap(input_tensor, class_idx=class_idx)

        for layer_name, heatmap in heatmaps.items():
            if save_path:
                gradcam_viewer.save_heatmap(img_np, heatmap, save_path)
            else:
                overlayed = gradcam_viewer.overlay_heatmap(img_np, heatmap)

        return class_idx, class_name, confidence_score


def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Image Prediction')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory of images')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generate Grad-CAM overlay')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output JSON file for batch predictions')
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error('Either --image or --image_dir must be provided')

    config = Config(args.config)
    if args.ensemble:
        config.cfg.model.use_ensemble = True

    predictor = Predictor(config, args.checkpoint)

    if args.image:
        if args.gradcam:
            gradcam_path = args.image.replace('.', '_gradcam.')
            class_idx, class_name, confidence = predictor.predict_with_gradcam(
                args.image, save_path=gradcam_path
            )
            print(f'Grad-CAM saved to: {gradcam_path}')
        else:
            class_idx, class_name, confidence = predictor.predict(args.image)

        print(f'Image: {args.image}')
        print(f'Prediction: {class_name} (class {class_idx})')
        print(f'Confidence: {confidence:.4f}')

    if args.image_dir:
        import glob
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.pgm', '*.dcm']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))

        if not image_paths:
            print(f'No images found in {args.image_dir}')
            return

        results = predictor.predict_batch(image_paths)
        print(f'\nProcessed {len(results)} images:')

        benign_count = sum(1 for _, name, _ in results if name == 'benign')
        malignant_count = sum(1 for _, name, _ in results if name == 'malignant')
        print(f'  Benign: {benign_count}')
        print(f'  Malignant: {malignant_count}')

        output_data = []
        for path, class_name, confidence in results:
            output_data.append({
                'image_path': path,
                'prediction': class_name,
                'confidence': round(confidence, 4)
            })
            print(f'  {os.path.basename(path):40s} -> {class_name:10s} (conf: {confidence:.4f})')

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
