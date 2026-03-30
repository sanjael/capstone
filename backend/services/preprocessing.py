"""
services/preprocessing.py — PRODUCTION UPGRADE v3
────────────────────────────────────────────────────
Fix: Inference normalization NOW matches training validation transform.
     Training val_tf used: Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
     Previous inference pipeline had NO normalization — silent distribution shift fixed.

Pipeline:
  1. Resize to target size (LANCZOS)
  2. CLAHE contrast enhancement
  3. Skull strip simulation
  4. ToTensor() + ImageNet normalization
  5. TTA: 3 views (original, h-flip, brightness shift) — all normalized
"""

import numpy as np
import torch
import cv2
from PIL import Image
from io import BytesIO
from torchvision import transforms
from typing import Tuple, List


# ── ImageNet statistics (must match training val_tf) ──────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_SIZE_PRIMARY   = 224
_SIZE_SECONDARY = 299

# ── Transform definitions ──────────────────────────────────────────────────────
# FIX: All inference transforms now include ImageNet normalization to match
#      the validation loader used during training.

_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_tta_flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_tta_bright_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ── CLAHE ─────────────────────────────────────────────────────────────────────

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the L channel of LAB colorspace.
    Enhances local contrast without amplifying noise — improves tumor boundary visibility.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


# ── Skull Strip Simulation ─────────────────────────────────────────────────────

def skull_strip_simulation(img_rgb: np.ndarray) -> np.ndarray:
    """
    Approximate skull stripping by masking the largest bright contour region.
    Reduces skull/bone features that confuse the CNN.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=2)
    result = img_rgb.copy()
    result[mask == 0] = 0
    return result


# ── Image Loading ──────────────────────────────────────────────────────────────

def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load PIL RGB image from raw bytes."""
    return Image.open(BytesIO(data)).convert("RGB")


# ── Primary Preprocessing ──────────────────────────────────────────────────────

def _enhance(img: Image.Image, size: int) -> Tuple[np.ndarray, Image.Image]:
    """
    Resize → CLAHE → Skull strip.
    Returns (enhanced_np_uint8, enhanced_PIL).
    """
    img_resized = img.resize((size, size), Image.LANCZOS)
    np_raw = np.array(img_resized, dtype=np.uint8)
    np_enhanced = apply_clahe(np_raw)
    np_enhanced = skull_strip_simulation(np_enhanced)
    pil_enhanced = Image.fromarray(np_enhanced)
    return np_enhanced, pil_enhanced


def preprocess_for_inference(
    img: Image.Image,
    size: int = _SIZE_PRIMARY,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Matches the training val_tf exactly:
      - Resize to size x size (LANCZOS)
      - ToTensor()
      - Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    Returns:
        tensor : (1, 3, size, size) float32 normalized
        np_img : (size, size, 3) uint8 for CAM overlay
    """
    img_resized = img.resize((size, size), Image.LANCZOS)
    np_img = np.array(img_resized, dtype=np.uint8)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img_resized).unsqueeze(0)
    return tensor, np_img


# ── TTA Preprocessing ─────────────────────────────────────────────────────────

def preprocess_tta(
    img: Image.Image,
    size: int = _SIZE_PRIMARY,
) -> List[torch.Tensor]:
    """
    Returns 3 normalized tensors for Test-Time Augmentation:
        [original, horizontal flip, brightness shift]
    All tensors are (1, 3, size, size) ImageNet-normalized.
    """
    np_enhanced, pil_enhanced = _enhance(img, size)

    t_orig   = _base_transform(pil_enhanced).unsqueeze(0)
    t_flip   = _tta_flip_transform(pil_enhanced).unsqueeze(0)
    t_bright = _tta_bright_transform(pil_enhanced).unsqueeze(0)

    return [t_orig, t_flip, t_bright]


# ── Multi-scale (optional, for ensemble inference) ────────────────────────────

def get_multi_scale_tensors(
    img: Image.Image,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns normalized tensors at 224×224 and 299×299, plus the 224 numpy for CAM.
    """
    t224, np224 = preprocess_for_inference(img, size=_SIZE_PRIMARY)
    t299, _     = preprocess_for_inference(img, size=_SIZE_SECONDARY)
    return t224, t299, np224
