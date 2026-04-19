"""
utils/gradcam.py — Gradient-weighted Class Activation Mapping
Highlights regions of the face the model focused on to make its decision.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class GradCAM:
    """
    Hooks into the last convolutional block of the model to capture:
      - Activations (forward hook)
      - Gradients  (backward hook)
    Then computes the weighted activation map.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        # Default: last conv block of EfficientNet-B4
        if target_layer is None:
            target_layer = model.base.features[-1]

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        """
        x: tensor of shape (1, 3, H, W)
        Returns: cam (numpy array, shape H x W, values 0-1)
        """
        self.model.eval()
        output = self.model(x)            # forward
        self.model.zero_grad()
        output.backward()                 # backward w.r.t. fake class score

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


def generate_gradcam(model, image_path: str, output_path: str) -> bool:
    """
    Generate a Grad-CAM heatmap overlay and save it.

    Args:
        model:        The EfficientNetDetector instance
        image_path:   Path to input image
        output_path:  Path to save the overlay image

    Returns:
        True on success, False on failure
    """
    try:
        device = next(model.parameters()).device

        # Load and preprocess
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return False

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(img_rgb)
        tensor   = TRANSFORM(pil_img).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        # Compute CAM
        gradcam = GradCAM(model)
        cam = gradcam(tensor)

        # Resize CAM to match original image
        h, w = img_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Convert CAM to heatmap (jet colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on original image (60% original, 40% heatmap)
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

        # Save
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return True

    except Exception as e:
        print(f"[GradCAM] Error: {e}")
        return False
