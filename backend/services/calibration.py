"""
services/calibration.py — PRODUCTION UPGRADE v3
─────────────────────────────────────────────────
Changes from v2:
  - calibrate_classification_probs() accepts override_temperature parameter
    so predictor can inject the file-loaded temperature at call time.
  - fit_temperature() utility added: can be run offline on a validation set
    to produce models/calibration.json without retraining.
  - ECE computation unchanged (correct).
  - Adaptive temperature logic retained but now uses override when provided.

Usage (offline calibration fitting — no GPU needed):
    from services.calibration import fit_temperature_from_val
    fit_temperature_from_val(model, val_loader, device="cpu", out_path="models/calibration.json")
"""

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 2.0
SMOOTHING_EPSILON   = 0.05
CUS_LOW             = 0.05
CUS_MODERATE        = 0.10
HIGH_TEMPERATURE    = 4.0


# ── Adaptive temperature (CUS-based) ──────────────────────────────────────────

def get_adaptive_temperature(cus: float, base_temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    Returns calibration temperature based on MC Dropout uncertainty (CUS).
    If a base_temperature is provided (from file), it replaces DEFAULT_TEMPERATURE.
    """
    if cus <= CUS_LOW:
        return base_temperature
    if cus >= CUS_MODERATE:
        return HIGH_TEMPERATURE
    ratio = (cus - CUS_LOW) / (CUS_MODERATE - CUS_LOW)
    return base_temperature + ratio * (HIGH_TEMPERATURE - base_temperature)


# ── Temperature scaling ────────────────────────────────────────────────────────

def temperature_scale_logits(
    logits: torch.Tensor,
    temperature: float,
) -> np.ndarray:
    """
    Apply temperature scaling to raw logits, return calibrated softmax probabilities.

    Args:
        logits      : raw model output, shape (1, num_classes)
        temperature : positive scaling factor

    Returns:
        numpy array of shape (num_classes,)
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    scaled = logits.float() / temperature
    probs = F.softmax(scaled, dim=1)
    return probs.detach().cpu().numpy()[0]


# ── Label smoothing ────────────────────────────────────────────────────────────

def smooth_probabilities(
    probs: np.ndarray,
    epsilon: float = SMOOTHING_EPSILON,
) -> np.ndarray:
    """
    Apply label smoothing: p_smooth = (1 - ε) * p + ε / K
    Prevents overconfident 0.0/1.0 outputs.
    """
    K = len(probs)
    smoothed = (1.0 - epsilon) * probs + epsilon / K
    smoothed = smoothed / smoothed.sum()
    return smoothed.astype(np.float32)


# ── Main calibration entry point ───────────────────────────────────────────────

def calibrate_classification_probs(
    logits: torch.Tensor,
    cus: float,
    epsilon: float = SMOOTHING_EPSILON,
    override_temperature: Optional[float] = None,
) -> np.ndarray:
    """
    Full calibration pipeline: temperature scaling → label smoothing.

    Args:
        logits               : raw logits from classification model, shape (1, C)
        cus                  : Cascade Uncertainty Score from MC Dropout
        epsilon              : label smoothing factor
        override_temperature : if provided, uses this as the base temperature
                               instead of DEFAULT_TEMPERATURE. Pass the value
                               loaded from models/calibration.json.

    Returns:
        calibrated + smoothed probability vector, shape (C,)
    """
    base_T = override_temperature if override_temperature is not None else DEFAULT_TEMPERATURE
    T = get_adaptive_temperature(cus, base_temperature=base_T)
    probs = temperature_scale_logits(logits, T)
    probs = smooth_probabilities(probs, epsilon)
    return probs


# ── Offline calibration fitting (no retraining) ────────────────────────────────

def fit_temperature_from_val(
    model: torch.nn.Module,
    val_loader,
    device: str = "cpu",
    out_path: str = "models/calibration.json",
    num_classes: int = 3,
) -> float:
    """
    Fit optimal temperature on a validation DataLoader using NLL minimization.
    Saves result to out_path as JSON.

    This requires ONLY a forward pass (no retraining, no GPU needed).
    Run once offline, then the predictor loads the result automatically.

    Args:
        model      : trained classification model (in eval mode)
        val_loader : DataLoader yielding (images, labels)
        device     : "cpu" or "cuda"
        out_path   : path to save calibration.json
        num_classes: number of output classes

    Returns:
        optimal temperature (float)
    """
    from scipy.optimize import minimize_scalar

    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)  # (N, C)
    labels_cat = torch.cat(all_labels, dim=0)  # (N,)

    def nll_loss(T: float) -> float:
        if T <= 0:
            return 1e9
        scaled = logits_cat / T
        log_probs = F.log_softmax(scaled, dim=1)
        nll = F.nll_loss(log_probs, labels_cat)
        return float(nll.item())

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
    optimal_T = float(result.x)

    logger.info(f"Optimal calibration temperature: {optimal_T:.4f}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"temperature": optimal_T}, f, indent=2)
    logger.info(f"Saved calibration temperature to {out_path}")

    return optimal_T


# ── ECE evaluation ─────────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        confidences : max predicted probability per sample, shape (N,)
        accuracies  : binary correct/incorrect per sample, shape (N,)
        n_bins      : number of equal-width bins

    Returns:
        ECE (float) — lower is better, 0.0 = perfect calibration
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc  = float(accuracies[mask].mean())
        ece += (mask.sum() / N) * abs(bin_conf - bin_acc)

    return float(ece)
