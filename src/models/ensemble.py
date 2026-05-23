import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .model_factory import ModelFactory


class EnsembleModel(nn.Module):
    def __init__(
        self,
        models_dict: Dict[str, nn.Module],
        num_classes: int,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models_dict.values())
        self.model_names = list(models_dict.keys())
        self.num_classes = num_classes

        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(out)

        stack = torch.stack(outputs, dim=0)
        weighted = stack * self.weights.view(-1, 1, 1)
        return weighted.sum(dim=0)

    def get_individual_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = {}
        for name, model in zip(self.model_names, self.models):
            result[name] = model(x)
        return result


def create_ensemble(config) -> EnsembleModel:
    models_list = ModelFactory.create_all_ensemble_models(config)
    models_dict = {name: m for name, m in models_list}
    return EnsembleModel(models_dict, num_classes=config.model.num_classes)
