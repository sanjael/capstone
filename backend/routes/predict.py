# routes/predict.py — PRODUCTION UPGRADE v3
# Updated response model to include all new fields:
#   tumor_region, cam_reliability, tta_agreement, is_ood, ood_reason

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any
import logging

from config import get_settings
from services.predictor import run_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Sub-models ─────────────────────────────────────────────────────────────────

class UncertaintyProfile(BaseModel):
    aleatoric: float
    epistemic: float
    dominant: str


class TumorRegion(BaseModel):
    area_percent: float
    centroid_x_norm: float
    centroid_y_norm: float
    region_confidence: float
    quadrant: str


class CAMReliability(BaseModel):
    recommended_cam: str
    reasoning: str
    agreement_score: Optional[float] = None


# ── Main response ──────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    tumor_detected: bool
    decision_type: str
    tumor_type: Optional[str] = None
    confidence: float
    uncertainty: float
    prediction_entropy: Optional[float] = None
    uncertainty_profile: Optional[UncertaintyProfile] = None
    reliability: str
    all_class_probs: Optional[Dict[str, float]] = None

    # Heatmaps
    heatmap_image: Optional[str] = None
    heatmap_gradcam: Optional[str] = None
    heatmap_eigencam: Optional[str] = None
    heatmap_scorecam: Optional[str] = None

    # Novelty outputs
    tumor_region: Optional[TumorRegion] = None
    cam_reliability: Optional[CAMReliability] = None
    tta_agreement: Optional[float] = None
    tta_agreement_pct: Optional[float] = None

    # OOD
    is_ood: bool = False
    ood_reason: Optional[str] = None

    # Risk
    risk_level: Optional[str] = None
    risk_color: Optional[str] = None
    clinical_note: Optional[str] = None
    recommendation: Optional[str] = None

    error: Optional[str] = None


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, BMP, etc.).",
        )

    image_bytes = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed: {settings.MAX_FILE_SIZE_MB} MB.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    logger.info(f"Received image: {file.filename}, size={len(image_bytes)} bytes")

    result = await run_prediction(image_bytes)
    return PredictionResponse(**result)
