import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from ..config import Config


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ModelFactory:
    @staticmethod
    def create(model_name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        if model_name == 'cnn':
            return CustomCNN(num_classes=num_classes, dropout=dropout)

        if model_name == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b3(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b4(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights, memory_efficient=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    @staticmethod
    def create_all_ensemble_models(config: Config):
        models_list = []
        model_names = config.model.ensemble_models
        for name in model_names:
            m = ModelFactory.create(
                model_name=name,
                num_classes=config.model.num_classes,
                pretrained=config.model.pretrained,
                dropout=config.model.dropout
            )
            models_list.append((name, m))
        return models_list


def create_model(config: Config) -> nn.Module:
    return ModelFactory.create(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout
    )
