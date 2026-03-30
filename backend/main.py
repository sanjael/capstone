"""
main.py — FastAPI application entry point
─────────────────────────────────────────
Run with:
  uvicorn main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from routes.predict import router as predict_router
from routes.report import router as report_router
from routes.analytics import router as analytics_router

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("Brain Tumor Detection System starting up...")
    # Trigger model loading on startup (lazy loading also supported)
    try:
        from services.model_loader import get_detection_model, get_classification_model
        get_detection_model()
        get_classification_model()
        logger.info("Models preloaded successfully.")
    except Exception as e:
        logger.warning(f"Model preloading failed (will retry on first request): {e}")

    logger.info("System ready.")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Shutting down...")


app = FastAPI(
    title="Brain Tumor Detection & Classification API",
    description=(
        "Advanced Explainable AI System — ResNet101 + Grad-CAM++ + Bayesian Uncertainty. "
        "Public tumor detection and analysis. No authentication required."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict_router)
app.include_router(report_router)
app.include_router(analytics_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "Brain Tumor Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "features": ["prediction", "grad-cam", "risk-analysis", "analytics", "pdf-reports"],
    }


@app.get("/health", tags=["Health"])
async def health():
    from services.model_loader import models_loaded, get_device
    return {
        "status": "healthy",
        "models_loaded": models_loaded(),
        "device": get_device(),
    }
