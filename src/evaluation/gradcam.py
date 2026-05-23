import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Dict, List, Tuple
import os


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        b, c, h, w = x.size()

        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class GradCAMView:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[str],
        device: torch.device
    ):
        self.model = model
        self.device = device
        self.gradcams = {layer: GradCAM(model, layer) for layer in target_layers}
        if not target_layers:
            self.target_layers = self._find_last_conv()
            self.gradcams = {l: GradCAM(model, l) for l in self.target_layers}
        else:
            self.target_layers = target_layers

    def _find_last_conv(self) -> List[str]:
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                last_conv = name
        if last_conv is None:
            last_conv = 'features' if hasattr(self.model, 'features') else 'conv1'
        return [last_conv]

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        self.model.eval()
        results = {}
        for layer_name, gradcam in self.gradcams.items():
            cam = gradcam.generate(image_tensor, class_idx)
            results[layer_name] = cam
        return results

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlayed

    def save_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        save_path: str,
        alpha: float = 0.5
    ):
        overlayed = self.overlay_heatmap(image, heatmap, alpha)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(overlayed).save(save_path)
