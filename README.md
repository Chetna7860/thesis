# Breast Cancer Classification using Deep Learning

A comprehensive thesis-level breast cancer classification system using multiple deep learning architectures with transfer learning, ensemble methods, and explainable AI (Grad-CAM).

## Datasets Supported

| Dataset | Type | Images | Classes |
|---------|------|--------|---------|
| **BreaKHis** | Histopathology (H&E stained) | 7,909 | Benign, Malignant |
| **INbreast** | Mammography (DICOM) | 410 | Normal, Abnormal |
| **MIAS** | Mammography (PGM) | 322 | Normal, Benign, Malignant |

## Features

- **Multi-dataset training** - Train on one or combine multiple datasets
- **Transfer Learning** - EfficientNet-B3/B4, DenseNet121, ResNet50 with pretrained ImageNet weights
- **Ensemble Learning** - Combine predictions from multiple architectures
- **Stratified K-Fold Cross-Validation** - Up to 10 folds
- **Advanced Augmentations** - Rotation, flip, brightness/contrast, gamma, noise, coarse dropout
- **CLAHE Preprocessing** - Adaptive histogram equalization for enhanced tissue contrast
- **MixUp / CutMix** - Advanced data augmentation techniques
- **Class Balancing** - Weighted random sampling for imbalanced datasets
- **Mixed Precision Training** - Faster training with NVIDIA AMP
- **Cosine Annealing LR** - Learning rate scheduling with warm restarts
- **Early Stopping** - Prevent overfitting with patience-based stopping
- **TensorBoard Logging** - Real-time training monitoring
- **Checkpointing** - Save best and epoch-level checkpoints
- **Grad-CAM** - Visual explanations of model decisions
- **Comprehensive Metrics** - Accuracy, Precision, Recall, F1, Sensitivity, Specificity, AUC-ROC, Kappa, MCC
- **Confusion Matrix & ROC Curves** - Automatic visualization

## Project Structure

```
breast_cancer_project/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── config.py                # Config loader
│   ├── datasets/
│   │   ├── base_dataset.py      # Base dataset class
│   │   ├── breakhis_dataset.py  # BreaKHis dataset loader
│   │   ├── inbreast_dataset.py  # INbreast dataset loader
│   │   ├── mias_dataset.py      # MIAS dataset loader
│   │   └── dataloader.py        # DataLoader factory
│   ├── models/
│   │   ├── model_factory.py     # Model creation (EfficientNet, DenseNet, ResNet)
│   │   └── ensemble.py          # Ensemble model
│   ├── preprocessing/
│   │   ├── augmentations.py     # Training/val/test transforms
│   │   ├── clahe.py             # CLAHE implementation
│   │   └── mixup_cutmix.py      # MixUp & CutMix
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   └── kfold.py             # K-fold cross validation
│   ├── evaluation/
│   │   ├── metrics.py           # Classification metrics
│   │   ├── gradcam.py           # Grad-CAM visualization
│   │   └── visualizer.py        # Plotting utilities
│   └── utils/
│       ├── early_stopping.py    # Early stopping
│       └── logger.py            # Logging (file + TensorBoard)
├── outputs/
│   ├── checkpoints/             # Model checkpoints
│   ├── logs/                    # Training logs
│   ├── plots/                   # Generated plots
│   ├── models/                  # Saved final models
│   └── gradcam/                 # Grad-CAM visualizations
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
├── predict.py                   # Inference entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Configuration

Edit `config/config.yaml` to set dataset paths and training parameters:

```yaml
data:
  breakhis:
    path: C:\path\to\BreaKHis_v1\histology_slides\breast
    use: true
    magnification: [40, 100, 200, 400]
  inbreast:
    path: C:\path\to\inbreast
    use: false
  mias:
    path: C:\path\to\mias
    use: false
```

### Training

```bash
# Train with EfficientNet-B3 on BreaKHis
python train.py --model efficientnet_b3 --epochs 100 --batch_size 32

# Train with ensemble of all models
python train.py --ensemble --epochs 80

# Train with specific datasets
python train.py --datasets breakhis --magnification 40 100 200 400

# Train with K-fold cross validation
python train.py --model densenet121 --kfold 5

# Custom learning rate and batch size
python train.py --model resnet50 --lr 0.0005 --batch_size 64

# Resume from checkpoint
python train.py --resume outputs/checkpoints/run_name/model_best.pth
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py --checkpoint outputs/checkpoints/run_name/model_best.pth

# Evaluate with Grad-CAM visualization
python evaluate.py --checkpoint path/to/model.pth --gradcam --num_samples 20

# Evaluate ensemble model
python evaluate.py --checkpoint path/to/model.pth --ensemble
```

### Inference

```bash
# Predict single image
python predict.py --checkpoint outputs/models/efficientnet_b3_final.pth --image path/to/image.png

# Predict with Grad-CAM overlay
python predict.py --checkpoint path/to/model.pth --image path/to/image.png --gradcam

# Batch predict directory
python predict.py --checkpoint path/to/model.pth --image_dir path/to/images/ --output results.json

# Ensemble prediction
python predict.py --checkpoint path/to/model.pth --image_dir path/to/images/ --ensemble
```

## Models

| Architecture | Param Count | Input Size | Top-1 ImageNet |
|-------------|-------------|------------|-----------------|
| EfficientNet-B3 | 12.3M | 300x300 | 81.7% |
| EfficientNet-B4 | 19.3M | 380x380 | 82.9% |
| DenseNet121 | 8.0M | 224x224 | 74.4% |
| ResNet50 | 25.6M | 224x224 | 76.1% |

## Metrics

The pipeline computes the following metrics automatically:
- **Accuracy**
- **Precision** (Positive Predictive Value)
- **Recall** (Sensitivity, True Positive Rate)
- **F1-Score** (Harmonic mean of Precision & Recall)
- **Specificity** (True Negative Rate)
- **AUC-ROC** (Area Under ROC Curve)
- **Cohen's Kappa** (Inter-rater agreement)
- **Matthews Correlation Coefficient (MCC)**

## Output Files

```
outputs/
├── checkpoints/
│   └── {run_name}/
│       ├── model_best.pth        # Best validation checkpoint
│       └── model_epoch_{n}.pth   # Epoch-level checkpoint
├── logs/
│   └── {run_name}/
│       ├── training.log          # Training log file
│       └── events.out.tfevents.* # TensorBoard events
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_history.png
│   ├── metrics_comparison.png
│   └── metrics_report.txt
├── models/
│   └── {model_name}_final.pth    # Final trained model
└── gradcam/
    └── *.png                     # Grad-CAM heatmaps
```

## Viewing TensorBoard

```bash
tensorboard --logdir outputs/logs
```

## License

This project is for research and educational purposes.
