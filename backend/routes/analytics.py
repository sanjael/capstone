"""
routes/analytics.py — Stateless analytics computed from submitted prediction results (no auth, no DB)
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class ScanEntry(BaseModel):
    tumor_detected: bool
    tumor_type: Optional[str] = None
    confidence: float
    risk_level: str


class AnalyticsRequest(BaseModel):
    scans: List[ScanEntry]


@router.post("/summary", summary="Compute analytics summary from a list of scan results")
def analytics_summary(body: AnalyticsRequest):
    scans = body.scans
    total = len(scans)
    if total == 0:
        return {"total": 0}

    detected = [s for s in scans if s.tumor_detected]
    type_dist: dict = {}
    for s in detected:
        if s.tumor_type:
            type_dist[s.tumor_type] = type_dist.get(s.tumor_type, 0) + 1

    avg_conf = round(sum(s.confidence for s in scans) / total, 4)

    risk_dist: dict = {}
    for s in scans:
        risk_dist[s.risk_level] = risk_dist.get(s.risk_level, 0) + 1

    return {
        "total_scans": total,
        "tumor_detected": len(detected),
        "no_tumor": total - len(detected),
        "detection_rate": round(len(detected) / total, 4),
        "average_confidence": avg_conf,
        "tumor_type_distribution": type_dist,
        "risk_distribution": risk_dist,
    }
