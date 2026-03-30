"""
routes/report.py — PDF report generation (no auth, no DB)
"""
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from services.report_generator import generate_pdf_report

router = APIRouter(prefix="/report", tags=["Report"])


class ReportRequest(BaseModel):
    user_name: str = "Clinician"
    tumor_detected: bool
    tumor_type: Optional[str] = None
    confidence: float
    uncertainty: float
    reliability: str
    risk_level: str
    clinical_note: str
    recommendation: str
    heatmap_image: Optional[str] = None


@router.post("/", summary="Generate PDF clinical report from prediction data")
def generate_report(body: ReportRequest):
    pdf_bytes = generate_pdf_report(
        user_name=body.user_name,
        prediction_data=body.model_dump(),
    )
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=neuroscan_report.pdf"},
    )
