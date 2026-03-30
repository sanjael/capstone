# services/predictor.py — PRODUCTION GRADE (ALL FIXES APPLIED)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import logging
import math
import json
from pathlib import Path

from config import get_settings
from services.risk_analysis import get_risk_report
from services.model_loader import (
    get_detection_model,
    get_classification_model,
    get_device
)
from services.preprocessing import preprocess_for_inference
from services.gradcam import GradCAMPlusPlus, EigenCAM

# ─────────────────────────────────────────────
settings = get_settings()
DEVICE   = get_device()
logger   = logging.getLogger(__name__)

logger.info(f"Predictor initialized on {DEVICE}")


# ─────────────────────────────────────────────
# STEP 6: Load detection threshold from threshold.json
# ─────────────────────────────────────────────
def _load_detection_threshold() -> float:
    threshold_path = Path(__file__).parent.parent / "models" / "threshold.json"
    try:
        if threshold_path.exists():
            data = json.loads(threshold_path.read_text())
            val  = float(data.get("threshold", 0.4))
            logger.info(f"Loaded detection threshold from threshold.json: {val}")
            return val
    except Exception as e:
        logger.warning(f"Could not load threshold.json, using default 0.4: {e}")
    return 0.4

_DETECTION_THRESHOLD = _load_detection_threshold()


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
def preprocess(image: Image.Image):
    tensor, np_processed = preprocess_for_inference(image, size=224)
    return tensor.to(DEVICE), np_processed


# ─────────────────────────────────────────────
# MC DROPOUT (DETECTION)
# ─────────────────────────────────────────────
def mc_dropout(model, x, passes=5):
    probs = []
    try:
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        for _ in range(passes):
            with torch.no_grad():
                if hasattr(model, "predict_proba"):
                    out = model.predict_proba(x)
                else:
                    out = torch.sigmoid(model(x))

            probs.append(float(out.detach().cpu().flatten()[0]))

        return float(np.mean(probs)), float(np.std(probs))

    finally:
        model.eval()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
async def run_prediction(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x, original_np = preprocess(image)

        detection_model     = get_detection_model()
        classification_model = get_classification_model()

        # ───────── DETECTION ─────────
        prob, uncertainty = mc_dropout(detection_model, x)

        # STEP 6: Use threshold loaded from threshold.json (not hardcoded)
        HIGH_THR = _DETECTION_THRESHOLD
        LOW_THR  = settings.LOW_THRESHOLD

        if prob >= HIGH_THR:
            tumor_detected = True
            status = "HIGH"
        elif prob <= LOW_THR:
            tumor_detected = False
            status = "HIGH"
        else:
            tumor_detected = True
            status = "LOW"

        # FN safety
        if prob > (HIGH_THR - 0.1) and uncertainty > 0.2:
            tumor_detected = True
            status = "LOW"

        # ───────── CLASSIFICATION ─────────
        tumor_type = None
        probs      = None
        max_prob   = 0.0
        entropy    = None

        if tumor_detected:
            try:
                # The Ensemble classification model returns PROBABILITIES (softmax already applied)
                # so we do not need to apply another softmax here.
                probs_tensor = classification_model(x)
                probs = probs_tensor.detach().cpu().numpy()[0]

                # Clamp to valid probability range for math safety
                probs = np.clip(probs, 1e-6, 1.0)
                probs = probs / probs.sum()

                classes   = ["glioma", "meningioma", "pituitary"]
                max_prob  = float(np.max(probs))
                predicted_class = classes[int(np.argmax(probs))]

                # ENTROPY (ALEATORIC UNCERTAINTY)
                entropy_raw = -np.sum(probs * np.log(probs))
                entropy     = float(entropy_raw / math.log(len(classes)))

                # Always assign the predicted class so the UI shows it
                tumor_type = predicted_class
                classification_uncertain = max_prob < settings.CLASSIFICATION_CONF_THRESHOLD

            except Exception as cls_err:
                logger.warning(f"Classification failed: {cls_err}")

        # ───────── RELIABILITY ─────────
        if status == "LOW":
            reliability = "LOW"
        elif max_prob < 0.6:
            reliability = "MEDIUM"
        else:
            reliability = "HIGH"

        # ───────── OUTPUT ─────────
        class_probs_dict = None
        if probs is not None:
            class_labels     = ["glioma", "meningioma", "pituitary"]
            class_probs_dict = {
                label: round(float(p), 6)
                for label, p in zip(class_labels, probs)
            }

        if status == "LOW":
            decision_type = "UNCERTAIN"
        elif reliability == "HIGH":
            decision_type = "CONFIDENT"
        else:
            decision_type = "FALLBACK"

        # ───────── EXPLAINABILITY ─────────
        heatmap_gradcam  = None
        heatmap_eigencam = None

        cam_model     = classification_model if tumor_detected else detection_model
        cam_class_idx = None

        if tumor_detected and tumor_type not in ("uncertain", None):
            cam_class_idx = ["glioma", "meningioma", "pituitary"].index(tumor_type)
            with torch.no_grad():
                raw_out    = classification_model(x)
                raw_argmax = int(raw_out.argmax(dim=1).item())
            if raw_argmax != cam_class_idx:
                cam_class_idx = raw_argmax

        try:
            gradcam = GradCAMPlusPlus(cam_model)
            heatmap_gradcam = gradcam.generate(x, original_np, class_idx=cam_class_idx)
            gradcam.remove_hooks()
        except Exception:
            pass

        try:
            eigencam = EigenCAM(cam_model)
            heatmap_eigencam = eigencam.generate(x, original_np, class_idx=cam_class_idx)
            eigencam.remove_hooks()
        except Exception:
            pass

        heatmap_image = heatmap_gradcam or heatmap_eigencam

        # ───────── RISK ─────────
        risk = get_risk_report(
            tumor_type=tumor_type if tumor_detected else None,
            confidence=prob,
            uncertainty=uncertainty,
            classification_uncertain=locals().get('classification_uncertain', False)
        )

        # ENTROPY OUTPUT
        prediction_entropy = round(float(entropy), 4) if entropy is not None else None

        # UNCERTAINTY PROFILE
        if prediction_entropy is not None:
            aleatoric = prediction_entropy
            epistemic = round(float(uncertainty), 4)
            dominant  = "aleatoric" if aleatoric > epistemic else "epistemic"
            uncertainty_profile = {
                "aleatoric": aleatoric,
                "epistemic": epistemic,
                "dominant":  dominant,
            }
        else:
            uncertainty_profile = None

        return {
            "tumor_detected":       bool(tumor_detected),
            "decision_type":        decision_type,
            "tumor_type":           tumor_type,
            "confidence":           round(float(prob), 6),
            "uncertainty":          round(float(uncertainty), 6),
            "prediction_entropy":   prediction_entropy,
            "uncertainty_profile":  uncertainty_profile,
            "reliability":          reliability,
            "all_class_probs":      class_probs_dict,
            "heatmap_image":        heatmap_image,
            "heatmap_gradcam":      heatmap_gradcam,
            "heatmap_eigencam":     heatmap_eigencam,
            "risk_level":           risk.risk_level,
            "risk_color":           risk.risk_color,
            "clinical_note":        risk.clinical_note,
            "recommendation":       risk.recommendation,
        }

    except Exception as e:
        logger.exception("Prediction pipeline failed")
        return {
            "tumor_detected":      False,
            "decision_type":       "ERROR",
            "tumor_type":          None,
            "confidence":          0.0,
            "uncertainty":         1.0,
            "prediction_entropy":  None,
            "uncertainty_profile": None,
            "reliability":         "LOW",
            "all_class_probs":     None,
            "error":               str(e),
        }