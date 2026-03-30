"""
services/risk_analysis.py  (UPGRADED v2)
──────────────────────────────────────────
Confidence-weighted clinical risk mapping with:
- Graded severity levels (Critical/High/Moderate/Low/None)
- Confidence penalty: if model is uncertain, risk is upgraded
- Detailed clinical staging notes per tumor type
- WHO grade references for each tumor type
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskReport:
    risk_level: str             # "Critical" | "High" | "Moderate" | "Low" | "None"
    risk_color: str             # UI: "red" | "orange" | "yellow" | "green" | "slate"
    clinical_note: str
    recommendation: str
    who_grade: Optional[str] = None
    urgency: str = "routine"    # "immediate" | "urgent" | "semi-urgent" | "routine"


# ── Detailed per-type knowledge base ─────────────────────────────────────────

_RISK_DB: dict[str, RiskReport] = {

    "glioma": RiskReport(
        risk_level="High",
        risk_color="red",
        who_grade="WHO Grade II–IV",
        urgency="urgent",
        clinical_note=(
            "Gliomas are primary brain tumours arising from glial (supportive) cells. "
            "Low-grade gliomas (WHO II–III) grow slowly but can transform to high-grade. "
            "WHO Grade IV Glioblastoma Multiforme (GBM) is the most aggressive primary brain tumour, "
            "with median survival of 14–16 months with standard treatment. "
            "Infiltrative nature means defined surgical margins are often impossible. "
            "Common symptoms: headaches, seizures, cognitive decline, focal neurological deficits."
        ),
        recommendation=(
            "1. IMMEDIATE neuro-oncology referral (within 24–72h).\n"
            "2. Contrast-enhanced MRI brain + spine (to assess extent).\n"
            "3. Neurosurgical evaluation for maximal safe resection vs. biopsy.\n"
            "4. Multidisciplinary tumour board (MDT) review.\n"
            "5. Molecular profiling: IDH1/2 mutation, MGMT methylation, 1p/19q co-deletion.\n"
            "6. Discuss adjuvant radiotherapy (60 Gy) + temozolomide chemotherapy.\n"
            "7. Genetic counselling if BRCA or Lynch syndrome is suspected."
        ),
    ),

    "meningioma": RiskReport(
        risk_level="Moderate",
        risk_color="orange",
        who_grade="WHO Grade I–III",
        urgency="semi-urgent",
        clinical_note=(
            "Meningiomas arise from the meninges (brain/spinal cord covering). "
            "~90% are WHO Grade I (benign), with excellent surgical outcomes. "
            "WHO Grade II (atypical) and Grade III (anaplastic) are aggressive and require adjuvant therapy. "
            "Can cause mass effect by compressing adjacent brain structures, cranial nerves, or the brainstem. "
            "Symptoms vary by location: visual loss, facial numbness, hearing loss, focal weakness, seizures. "
            "Incidental discovery is common — many are small and asymptomatic."
        ),
        recommendation=(
            "1. Neurosurgical consultation (within 1–2 weeks for symptomatic cases).\n"
            "2. Gadolinium-enhanced MRI with volumetric analysis.\n"
            "3. Serial MRI every 6–12 months for small, asymptomatic lesions (<3 cm).\n"
            "4. Surgical resection (Simpson Grade I resection) for symptomatic/growing lesions.\n"
            "5. Stereotactic radiosurgery (SRS/Gamma Knife) for small or surgically inaccessible tumours.\n"
            "6. Post-resection WHO grading by histopathology is essential.\n"
            "7. If Grade II/III: adjuvant radiotherapy to tumour bed."
        ),
    ),

    "pituitary": RiskReport(
        risk_level="Moderate",
        risk_color="yellow",
        who_grade="WHO Grade I (usually)",
        urgency="semi-urgent",
        clinical_note=(
            "Pituitary adenomas account for ~15% of all intracranial tumours. "
            "Most are benign (WHO Grade I) and arise from the anterior pituitary gland. "
            "Macroadenomas (>10 mm) can compress the optic chiasm → bitemporal hemianopia. "
            "Hormone-secreting adenomas: GH → acromegaly, ACTH → Cushing's disease, "
            "Prolactin → hyperprolactinaemia (amenorrhoea/galactorrhoea), TSH (rare). "
            "Non-secreting adenomas cause hypopituitarism via compression. "
            "Pituitary apoplexy (sudden infarction/haemorrhage) is a medical emergency."
        ),
        recommendation=(
            "1. Endocrinological evaluation with complete hormone panel: "
            "GH, IGF-1, ACTH, cortisol (am), prolactin, TSH, LH, FSH, testosterone/oestradiol.\n"
            "2. Ophthalmology referral: formal visual field testing (Humphrey perimetry).\n"
            "3. Dedicated pituitary MRI with fine cuts (3 mm slices) through the sella.\n"
            "4. For prolactinomas: dopamine agonist therapy (cabergoline) first-line.\n"
            "5. For GH/ACTH-secreting: transsphenoidal surgery is treatment of choice.\n"
            "6. For non-secreting macroadenomas: surgery if causing mass effect.\n"
            "7. Radiation (conventional or stereotactic) for residual/recurrent disease."
        ),
    ),
}


_NO_TUMOR: RiskReport = RiskReport(
    risk_level="None",
    risk_color="green",
    urgency="routine",
    who_grade=None,
    clinical_note=(
        "No intracranial tumour detected on this MRI scan. "
        "The scan appears within normal limits for tumour detection purposes. "
        "Note: this AI system is designed specifically for tumour detection; "
        "it does not assess for other pathologies (stroke, demyelination, infection)."
    ),
    recommendation=(
        "1. Routine clinical follow-up as indicated by presenting symptoms.\n"
        "2. If neurological symptoms persist, comprehensive neurological evaluation is advised.\n"
        "3. Consider additional MRI sequences (FLAIR, DWI, SWI) if clinically warranted.\n"
        "4. No immediate neurosurgical intervention indicated.\n"
        "5. Re-scan in 6–12 months if initially equivocal symptoms."
    ),
)


# ── Confidence-weighted risk upgrading ────────────────────────────────────────

def get_risk_report(
    tumor_type: Optional[str],
    confidence: float = 1.0,
    uncertainty: float = 0.0,
    classification_uncertain: bool = False,
) -> RiskReport:
    """
    Returns a risk report, potentially upgraded based on model confidence.
    
    If the model is uncertain (high uncertainty or low confidence), the risk
    level is upgraded to ensure the clinician performs manual review.
    """
    # ── STEP 5 FIX: Uncertain tumor classification → ALWAYS High Risk ──────────
    if tumor_type and (tumor_type.lower() == "uncertain" or classification_uncertain):
        uncertain_note = (
            f"⚠️ TUMOR DETECTED — {tumor_type.upper()} (UNCERTAIN): The AI model detected a tumor and suspects {tumor_type}, but could not "
            "confidently verify the classification. This requires mandatory expert radiological review "
            "before any clinical decision is made. Do not treat this as a low-risk finding."
        )
        uncertain_rec = (
            "1. IMMEDIATE referral to neuro-oncology or radiology (within 24h).\n"
            "2. Contrast-enhanced MRI brain with dedicated sequences.\n"
            "3. Neurosurgical consultation for biopsy consideration.\n"
            "4. Multidisciplinary tumour board review.\n"
            "5. Do NOT delay based on AI uncertainty — treat as High Risk."
        )
        return RiskReport(
            risk_level="High",
            risk_color="red",
            clinical_note=uncertain_note,
            recommendation=uncertain_rec,
            who_grade=None,
            urgency="urgent",
        )
    # ──────────────────────────────────────────────────────────────────

    if not tumor_type:
        return _NO_TUMOR

    base = _RISK_DB.get(tumor_type.lower(), _NO_TUMOR)

    # If model is uncertain, warn in the clinical note
    if uncertainty > 0.15 or confidence < 0.70:
        upgraded_note = (
            f"⚠️ MODEL UNCERTAINTY ALERT: Confidence = {confidence*100:.1f}%, "
            f"MC Uncertainty = ±{uncertainty:.4f}. "
            "This prediction has significant uncertainty — mandatory expert review before any clinical decisions.\n\n"
            + base.clinical_note
        )
        upgraded_rec = (
            "PRIORITY ACTION: Due to AI model uncertainty, all recommendations below "
            "require neurology/radiology verification before implementation.\n\n"
            + base.recommendation
        )
        # Upgrade risk level for uncertain predictions
        risk_upgrade = {
            "None": "Low",
            "Low": "Moderate",
            "Moderate": "High",
            "High": "Critical",
            "Critical": "Critical",
        }
        color_upgrade = {
            "green": "yellow",
            "yellow": "orange",
            "orange": "red",
            "red": "red",
        }
        return RiskReport(
            risk_level=risk_upgrade.get(base.risk_level, base.risk_level),
            risk_color=color_upgrade.get(base.risk_color, base.risk_color),
            clinical_note=upgraded_note,
            recommendation=upgraded_rec,
            who_grade=base.who_grade,
            urgency="urgent" if base.urgency == "routine" else base.urgency,
        )

    return base
