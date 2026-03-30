# services/model_loader.py — PRODUCTION GRADE (ENSEMBLE ENABLED)

import os
import threading
import logging
import torch

from config import get_settings

from models.resnet_models import (
    build_resnet_detection,
    build_resnet_classification,
)

from models.advanced_models import (
    build_efficient_detection,
    build_efficient_classification,
    EnsembleDetectionModel,
    EnsembleClassificationModel,
)

# ─────────────────────────────────────────────
settings = get_settings()
logger   = logging.getLogger(__name__)

_lock = threading.Lock()
_detection_model     = None
_classification_model = None
_models_loaded       = False


# ─────────────────────────────────────────────
# DEVICE CONTROL
# ─────────────────────────────────────────────
def _get_device():
    if settings.DEVICE == "cpu":
        return "cpu"
    elif settings.DEVICE == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:  # "auto"
        return "cuda" if torch.cuda.is_available() else "cpu"


_device = _get_device()


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def _load_models():
    global _detection_model, _classification_model, _models_loaded

    # Absolute path relative to this file's location (backend/)
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eff_det_path = os.path.join(base_dir, settings.MODEL_EFF_DET_PATH)
    res_det_path = os.path.join(base_dir, settings.MODEL_RES_DET_PATH)
    res_cls_path = os.path.join(base_dir, settings.MODEL_RES_CLS_PATH)
    eff_cls_path = os.path.join(base_dir, settings.MODEL_EFF_CLS_PATH)

    print("[INFO] ─────────────── MODEL LOADER ───────────────")
    print(f"[INFO] Device          : {_device}")
    print(f"[INFO] eff_det_path    : {eff_det_path} (exists={os.path.exists(eff_det_path)})")
    print(f"[INFO] res_det_path    : {res_det_path} (exists={os.path.exists(res_det_path)})")
    print(f"[INFO] res_cls_path    : {res_cls_path} (exists={os.path.exists(res_cls_path)})")
    print(f"[INFO] eff_cls_path    : {eff_cls_path} (exists={os.path.exists(eff_cls_path)})")
    print(f"[INFO] ENSEMBLE_ENABLED: {settings.ENSEMBLE_ENABLED}")
    print("[INFO] ────────────────────────────────────────────")

    # =====================================================
    # DETECTION — Ensemble if both weights exist
    # =====================================================
    try:
        eff_det_exists = os.path.exists(eff_det_path)
        res_det_exists = os.path.exists(res_det_path)

        if settings.ENSEMBLE_ENABLED and eff_det_exists and res_det_exists:
            logger.info("🔀 Loading ENSEMBLE detection (EfficientNet + ResNet101)")
            eff_det = build_efficient_detection(eff_det_path, _device)
            res_det = build_resnet_detection(res_det_path, _device)
            _detection_model = EnsembleDetectionModel(eff_det, res_det)
            _detection_model.eval()
            logger.info("✅ Ensemble detection loaded")

        elif eff_det_exists:
            logger.info("Loading EfficientNet Detection (single model)")
            _detection_model = build_efficient_detection(eff_det_path, _device)

        elif res_det_exists:
            logger.info("Loading ResNet101 Detection (single model)")
            _detection_model = build_resnet_detection(res_det_path, _device)

        else:
            logger.warning("⚠️  No trained detection weights found — using pretrained backbone only")
            _detection_model = build_efficient_detection(None, _device)

    except Exception as e:
        logger.exception("Detection model failed to load")
        raise RuntimeError(f"Detection model error: {e}")

    # =====================================================
    # CLASSIFICATION — Ensemble if both weights exist
    # =====================================================
    try:
        res_cls_exists = os.path.exists(res_cls_path)
        eff_cls_exists = os.path.exists(eff_cls_path)

        if settings.ENSEMBLE_ENABLED and res_cls_exists and eff_cls_exists:
            logger.info("🔀 Loading ENSEMBLE classification (ResNet101 + EfficientNet)")
            res_cls = build_resnet_classification(res_cls_path, _device)
            eff_cls = build_efficient_classification(eff_cls_path, _device)
            _classification_model = EnsembleClassificationModel(res_cls, eff_cls)
            _classification_model.eval()
            logger.info("✅ Ensemble classification loaded")

        elif res_cls_exists:
            logger.info("Loading ResNet101 Classification (single model)")
            _classification_model = build_resnet_classification(res_cls_path, _device)

        elif eff_cls_exists:
            logger.info("Loading EfficientNet Classification (single model)")
            _classification_model = build_efficient_classification(eff_cls_path, _device)

        else:
            logger.warning("⚠️  No trained classification weights found — using pretrained backbone only")
            _classification_model = build_resnet_classification(None, _device)

    except Exception as e:
        logger.exception("Classification model failed to load")
        raise RuntimeError(f"Classification model error: {e}")

    _models_loaded = True
    logger.info(f"✅ All models loaded successfully on [{_device}]")


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def get_detection_model():
    global _detection_model
    if _detection_model is None:
        with _lock:
            if _detection_model is None:
                _load_models()
    return _detection_model


def get_classification_model():
    global _classification_model
    if _classification_model is None:
        with _lock:
            if _classification_model is None:
                _load_models()
    return _classification_model


def get_device():
    return _device


def models_loaded() -> bool:
    return _models_loaded


# ─────────────────────────────────────────────
# RELOAD SUPPORT
# ─────────────────────────────────────────────
def reload_models():
    global _detection_model, _classification_model, _models_loaded

    with _lock:
        logger.info("Reloading models...")
        _detection_model      = None
        _classification_model = None
        _models_loaded        = False
        _load_models()

    logger.info("Reload complete")