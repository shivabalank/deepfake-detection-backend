"""
utils/gradcam.py
------------------
Grad-CAM implementation targeting the last convolutional block of the
EfficientNet-B4 backbone. Produces a heatmap highlighting the image
regions that most influenced the Real/Fake prediction.
"""

from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Args:
            input_tensor: normalized tensor of shape (1, 3, H, W)
        Returns:
            (cam_heatmap [H,W] float32 in [0,1], sigmoid_confidence)
        """
        self.model.zero_grad()
        logit = self.model(input_tensor)         # (1,)
        prob = torch.sigmoid(logit)
        prob.backward()

        gradients = self.gradients[0]             # (C, h, w)
        activations = self.activations[0]         # (C, h, w)

        weights = gradients.mean(dim=(1, 2))       # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32,
                           device=activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.cpu().numpy()
        h, w = input_tensor.shape[2:]
        cam_resized = cv2.resize(cam_np, (w, h))
        return cam_resized, float(prob.item())


def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays a Grad-CAM heatmap onto the original RGB image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay
