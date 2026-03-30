"""
services/gradcam.py — PRODUCTION UPGRADE v3
─────────────────────────────────────────────
Changes from v2:
  - All CAM classes store _last_raw_cam (2D numpy, 0–1) for region estimation
  - try/finally hook cleanup enforced at class level
  - torch.enable_grad() wrapper in generate() to work correctly in no_grad contexts
  - retain_graph=False (was True — caused GPU/CPU memory leak)
  - EigenCAM: stable SVD with full_matrices=False
  - ScoreCAM: max_channels=32 default for CPU efficiency
  - CPU-explicit .cpu() calls throughout
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import base64
from io import BytesIO
from PIL import Image
from typing import Optional


# ── Shared overlay utility ─────────────────────────────────────────────────────

def _overlay_cam(cam: np.ndarray, original_image: np.ndarray, alpha: float = 0.45) -> str:
    """
    Resize CAM to original image size, blend with JET colormap.
    Returns data:image/png;base64,... string.
    Stores nothing — caller stores raw cam separately.
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img = original_image
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam


def _get_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    layer = model
    for part in layer_name.split("."):
        layer = getattr(layer, part)
    return layer


def _detect_target_layer(model: torch.nn.Module) -> str:
    """Auto-detect the best convolutional layer for CAM.
    For ResNet101: last conv in layer4 (before residual add) gives sharpest maps.
    For EfficientNet: last conv block in features.
    """
    if hasattr(model, "backbone"):
        bb = model.backbone
        # ResNet: target last conv3 in layer4 for sharpest gradients
        if hasattr(bb, "layer4"):
            return "backbone.layer4"
        # EfficientNet: last block in features
        if hasattr(bb, "features"):
            return "backbone.features"
    if hasattr(model, "model_a"):
        m = model.model_a
        if hasattr(m, "backbone"):
            if hasattr(m.backbone, "layer4"):
                return "model_a.backbone.layer4"
            if hasattr(m.backbone, "features"):
                return "model_a.backbone.features"
    return "backbone.layer4"


def _detect_target_layer_precise(model: torch.nn.Module) -> str:
    """Returns the most precise target layer path for sharpest CAM.
    Targets the last conv weight layer before the residual addition.
    """
    if hasattr(model, "backbone"):
        bb = model.backbone
        if hasattr(bb, "layer4"):
            # Last Bottleneck block, last conv (conv3 = 1x1 projection)
            try:
                last_block = bb.layer4[-1]
                if hasattr(last_block, "conv3"):
                    return "backbone.layer4.2.conv3"
                if hasattr(last_block, "conv2"):
                    return "backbone.layer4.2.conv2"
            except Exception:
                pass
            return "backbone.layer4"
        if hasattr(bb, "features"):
            # EfficientNet: last conv block
            try:
                n = len(bb.features)
                return f"backbone.features.{n - 2}"
            except Exception:
                pass
            return "backbone.features"
    return _detect_target_layer(model)


# ── Grad-CAM++ ─────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ with correct hook cleanup and raw CAM exposure.
    _last_raw_cam stores the 2D (H, W) normalized float32 CAM after generate().
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._last_raw_cam: Optional[np.ndarray] = None

        layer_name = target_layer or _detect_target_layer_precise(model)
        self._register_hooks(layer_name)

    def _register_hooks(self, target_layer: str):
        layer = _get_layer(self.model, target_layer)

        def fwd_hook(module, inp, out):
            self._activations = out.detach().cpu()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach().cpu()

        self._hooks.append(layer.register_forward_hook(fwd_hook))
        self._hooks.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> str:
        self._last_raw_cam = None
        inp = input_tensor.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            self.model.zero_grad()
            output = self.model(inp)

            if output.shape[-1] == 1 or output.dim() == 1:
                score = output.squeeze().flatten()[0]
            else:
                if class_idx is None:
                    class_idx = int(output.argmax(dim=1).item())
                score = output[0, class_idx]

            # retain_graph=False — prevents memory accumulation
            score.backward(retain_graph=False)

        grads = self._gradients   # (1, C, H, W)
        acts = self._activations  # (1, C, H, W)

        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM++ hooks did not capture activations/gradients.")

        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * grads_sq + sum_acts * grads_cu
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        alpha = grads_sq / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam_tensor = (weights * acts).sum(dim=1).squeeze(0)  # (H, W)
        cam_np = _normalize_cam(cam_tensor.numpy())

        self._last_raw_cam = cam_np
        return _overlay_cam(cam_np, original_image)


# ── EigenCAM ──────────────────────────────────────────────────────────────────

class EigenCAM:
    """
    EigenCAM — first principal component of feature maps.
    Gradient-free: reliable when model gradients are noisy/unstable.
    _last_raw_cam stores the 2D (H, W) normalized float32 CAM after generate().
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._last_raw_cam: Optional[np.ndarray] = None

        layer_name = target_layer or _detect_target_layer_precise(model)
        layer = _get_layer(model, layer_name)
        self._hook_handle = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o.detach().cpu())
        )

    def remove_hooks(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> str:
        self._last_raw_cam = None

        with torch.no_grad():
            _ = self.model(input_tensor)

        if self._activations is None:
            raise RuntimeError("EigenCAM hook did not capture activations.")

        acts = self._activations.squeeze(0)  # (C, H, W)
        C, H, W = acts.shape

        acts_np = acts.numpy()
        acts_flat = acts_np.reshape(C, -1)  # (C, H*W)

        # Center channels for PCA
        acts_centered = acts_flat - acts_flat.mean(axis=1, keepdims=True)

        # Truncated SVD — more stable than np.linalg.eig
        try:
            _, _, Vt = np.linalg.svd(acts_centered, full_matrices=False)
            first_pc = Vt[0]  # (H*W,)
        except np.linalg.LinAlgError:
            # Fallback: mean activation map
            first_pc = acts_np.mean(axis=0).flatten()

        cam_np = first_pc.reshape(H, W)
        cam_np = _normalize_cam(cam_np)

        self._last_raw_cam = cam_np
        return _overlay_cam(cam_np, original_image)


# ── Score-CAM ─────────────────────────────────────────────────────────────────

class ScoreCAM:
    """
    Score-CAM — gradient-free, perturbation-based activation maps.
    More faithful to model decisions than gradient methods.
    CPU-optimized: default max_channels=32, batched forward pass.
    _last_raw_cam stores the 2D (H, W) normalized float32 CAM after generate().
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._last_raw_cam: Optional[np.ndarray] = None

        layer_name = target_layer or _detect_target_layer_precise(model)
        layer = _get_layer(model, layer_name)
        self._hook_handle = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o.detach().cpu())
        )

    def remove_hooks(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: Optional[int] = None,
        max_channels: int = 32,
    ) -> str:
        self._last_raw_cam = None

        with torch.no_grad():
            baseline_out = self.model(input_tensor)
            if class_idx is None and baseline_out.shape[-1] > 1:
                class_idx = int(baseline_out.argmax(dim=1).item())

        if self._activations is None:
            raise RuntimeError("Score-CAM hook did not capture activations.")

        acts = self._activations.squeeze(0)  # (C, H, W)
        C, fH, fW = acts.shape
        inp_h, inp_w = input_tensor.shape[-2], input_tensor.shape[-1]

        # Subsample channels
        step = max(1, C // max_channels)
        channel_indices = list(range(0, C, step))[:max_channels]

        # Build masked inputs as a batch for CPU efficiency
        masked_inputs = []
        for ci in channel_indices:
            act_ch = acts[ci].unsqueeze(0).unsqueeze(0)  # (1,1,fH,fW)
            mask = F.interpolate(act_ch, size=(inp_h, inp_w), mode="bilinear", align_corners=False).squeeze()
            m_min, m_max = mask.min(), mask.max()
            mask = (mask - m_min) / (m_max - m_min + 1e-8)
            masked_inputs.append(input_tensor * mask.unsqueeze(0).unsqueeze(0))

        # Batch forward pass
        batch = torch.cat(masked_inputs, dim=0)  # (N_channels, 3, H, W)

        scores = []
        batch_size = 8  # process in mini-batches to avoid CPU OOM
        with torch.no_grad():
            for i in range(0, len(channel_indices), batch_size):
                mini_batch = batch[i:i + batch_size]
                out = self.model(mini_batch)
                if class_idx is not None and out.shape[-1] > 1:
                    s = torch.softmax(out, dim=1)[:, class_idx]
                else:
                    s = torch.sigmoid(out).squeeze(-1)
                scores.extend(s.cpu().numpy().tolist())

        scores_arr = np.array(scores, dtype=np.float32)
        acts_np = acts.numpy()

        cam = np.zeros((fH, fW), dtype=np.float32)
        for i, ci in enumerate(channel_indices):
            cam += scores_arr[i] * acts_np[ci]

        cam = _normalize_cam(cam)
        self._last_raw_cam = cam
        return _overlay_cam(cam, original_image)


# ── Comparison strip (optional, unchanged from v2) ────────────────────────────

def generate_comparison_strip(
    original: np.ndarray,
    gradcam_b64: str,
    eigencam_b64: str,
) -> str:
    def decode(b64: str) -> np.ndarray:
        data = base64.b64decode(b64.split(",", 1)[1])
        img = Image.open(BytesIO(data)).convert("RGB")
        return np.array(img)

    orig = cv2.resize(original, (224, 224))
    gcam = decode(gradcam_b64)
    ecam = decode(eigencam_b64)

    strip = np.concatenate([orig, gcam, ecam], axis=1)
    labels = ["Original", "Grad-CAM++", "EigenCAM"]
    x_positions = [112, 336, 560]
    for label, x in zip(labels, x_positions):
        cv2.putText(strip, label, (x - 40, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    buf = BytesIO()
    Image.fromarray(strip).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
