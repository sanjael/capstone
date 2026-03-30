# 🧠 NeuroScan AI — Explainable Brain Tumor Detection System

> **Open, Production-Ready Clinical Decision-Support System for Brain Tumor Detection and Classification using Dual-Ensembles (EfficientNet-B4 + ResNet101) with Grad-CAM++, EigenCAM, MC Dropout Uncertainty Estimation, and Calibrated Temperature Scaling.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![No Auth](https://img.shields.io/badge/Auth-None%20(Open%20API)-brightgreen.svg)]()
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-orange.svg)](https://git-lfs.com)

---

## Overview

NeuroScan AI is a full-stack, open-access clinical decision-support system that:

- **Detects** whether an MRI scan contains a brain tumour (binary classification)
- **Classifies** the tumour type: Glioma, Meningioma, or Pituitary Adenoma
- **Explains** predictions visually using **Grad-CAM++** and **EigenCAM** heatmaps
- **Quantifies** uncertainty using **Monte Carlo Dropout** across 5 stochastic forward passes
- **Provides** WHO-graded clinical risk insights and follow-up recommendations
- **Generates** downloadable PDF clinical reports
- **Computes** stateless analytics from submitted scan results
- **Requires no login** — fully open API, no authentication, no database

---

## What Changed from v1

| Feature | v1 | v2 (Current) |
|---|---|---|
| Authentication | JWT + bcrypt | ❌ Removed — open API |
| Database | SQLAlchemy + SQLite | ❌ Removed — stateless |
| Detection Model | ResNet101 | ✅ EfficientNet-B4 |
| Classification Model | ResNet101 | ✅ ResNet101 (trained weights) |
| Explainability | Grad-CAM++ only | ✅ Grad-CAM++ + EigenCAM |
| Uncertainty | None | ✅ MC Dropout (5 passes) |
| Preprocessing | Basic resize + normalize | ✅ CLAHE + Skull Strip + TTA |
| PDF Reports | Auth-gated | ✅ Open endpoint |
| Analytics | DB-backed per-user | ✅ Stateless from request body |
| Training | Single train.py | ✅ Separate train_detection.py + train_classification.py |
| Architecture | Single Backbone per stage | ✅ Dual-Ensemble (EfficientNet + ResNet) deployed at both stages |
| Confidence Calibration | Raw Softmax | ✅ Temperature Scaling (T=1.5) before Softmax for realistic probabilities |
| Clinical Safety | Basic mapping | ✅ Mandatory "High Risk" constraint overrides for uncertain findings |

---

## System Architecture

```
Browser (React + Vite :3000)
        |
        |  REST / multipart  (no token required)
        v
FastAPI Backend (:8000)
   |-- POST /predict/          — MRI upload → full AI pipeline
   |-- POST /report/           — prediction data → PDF download
   +-- POST /analytics/summary — scan list → statistics
```

---

## AI Pipeline

```
MRI Upload (JPEG / PNG / BMP)
    |
    v
Advanced Preprocessing
  ├── Resize to 224×224 (LANCZOS)
  ├── CLAHE — local contrast enhancement (tumour boundary sharpening)
  ├── Skull Strip Simulation — removes bone/skull ring
  ├── ImageNet Normalization — mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
  └── TTA — 3 versions: original + h-flip + brightness shift
    |
    v
Tumor Detection (Ensemble: EfficientNet-B4 + ResNet101)
  ├── Dual Model Forward Pass → Averaged Probabilities
  ├── MC Dropout — 5 stochastic passes → mean probability + std uncertainty
  ├── HIGH_THR >= 0.30 → tumor_detected = True
  ├── LOW_THR  <= 0.15 → tumor_detected = False
  └── FN Safety — if prob > 0.3 and uncertainty > 0.2 → force True
    |
   / \
  /   \
No    Tumor Detected
Tumor   |
  |     v
  |   Tumor Classification (Ensemble: ResNet101 + EfficientNet-B4)
  |   ├── 3 classes: glioma / meningioma / pituitary
  |   ├── Temperature Scaling (T=1.5) applied strictly before Softmax
  |   └── Averaged confidence score outputs
  |     |
  |     v
  |   Reliability Assessment
  |   ├── status=LOW  → reliability = LOW  (uncertain zone)
  |   ├── cls_conf < 0.6 → reliability = MEDIUM
  |   └── cls_conf >= 0.6 → reliability = HIGH
  |     |
  |     v
  |   Multi-Method Explainability
  |   ├── Grad-CAM++  — gradient-weighted activation maps
  |   └── EigenCAM    — PCA on feature maps (gradient-free, stable)
  |     |
  |     v
  +-> WHO-Graded Risk Analysis
      ├── Uncertain  → HIGH RISK override (classification threshold exception)
      ├── Glioma     → High Risk    (WHO II–IV, urgent)
      ├── Meningioma → Moderate Risk (WHO I–III, semi-urgent)
      ├── Pituitary  → Moderate Risk (WHO I, hormonal)
      └── None       → No Risk      (routine follow-up)
          |
          v
      JSON Response + base64 Heatmap PNG
```

---

## Project Structure

```
brain_tumor_project/
|
|-- README.md
|-- .gitignore
|-- .gitattributes              # Git LFS tracking rules
|-- docker-compose.yml
|
|-- Training/                   # Raw dataset — tracked via Git LFS
|   |-- glioma/        (3,018 images)
|   |-- meningioma/    (2,183 images)
|   |-- notumor/       (1,945 images)
|   └── pituitary/     (2,504 images)
|
|-- Testing/                    # Raw dataset — tracked via Git LFS
|   |-- glioma/        (755 images)
|   |-- meningioma/    (546 images)
|   |-- notumor/       (487 images)
|   └── pituitary/     (626 images)
|
|-- backend/
|   |-- main.py                 # FastAPI app + CORS + GZip + lifespan
|   |-- config.py               # Pydantic settings (model paths, thresholds)
|   |-- requirements.txt
|   |-- setup_dataset.py        # Raw → train/val/test split builder
|   |-- train_detection.py      # EfficientNet-B4 detection training
|   |-- train_classification.py # ResNet101 classification training
|   |-- generate_graphs.py      # Training metrics visualisation
|   |-- Dockerfile
|   |-- .env
|   |
|   |-- models/
|   |   |-- advanced_models.py          # EfficientNet-B4 detection model
|   |   |-- resnet_models.py            # ResNet101 classification model + aliases
|   |   |-- detection_efficientnet.pth  # Trained detection weights (Git LFS)
|   |   |-- classification_model.pth    # Trained classification weights (Git LFS)
|   |   |-- metrics.json                # Detection training metrics
|   |   |-- classification_metrics.json # Classification training metrics
|   |   └── *.png                       # Training curves + confusion matrices
|   |
|   |-- routes/
|   |   |-- predict.py          # POST /predict/
|   |   |-- report.py           # POST /report/
|   |   └── analytics.py        # POST /analytics/summary
|   |
|   └── services/
|       |-- preprocessing.py    # CLAHE + skull strip + TTA + normalization
|       |-- gradcam.py          # Grad-CAM++ + EigenCAM
|       |-- model_loader.py     # Thread-safe singleton model loader
|       |-- predictor.py        # Full pipeline orchestration + MC Dropout
|       |-- risk_analysis.py    # WHO-graded risk + confidence upgrading
|       └── report_generator.py # FPDF2 clinical PDF generation
|
└── frontend/
    |-- index.html
    |-- vite.config.js
    |-- tailwind.config.js
    |-- package.json
    |
    └── src/
        |-- main.jsx
        |-- App.jsx
        |-- index.css
        |
        |-- components/
        |   |-- Navbar.jsx
        |   |-- ConfidenceBar.jsx
        |   |-- RiskBadge.jsx
        |   └── ResultsPanel.jsx
        |
        |-- pages/
        |   └── DashboardPage.jsx
        |
        └── services/
            └── api.js          # Axios instance — predict, report, analytics
```

---

## Dataset

### Raw Dataset (Training/ + Testing/)

| Folder | Class | Count |
|---|---|---|
| Training | glioma | 3,018 |
| Training | meningioma | 2,183 |
| Training | notumor | 1,945 |
| Training | pituitary | 2,504 |
| **Training Total** | | **9,650** |
| Testing | glioma | 755 |
| Testing | meningioma | 546 |
| Testing | notumor | 487 |
| Testing | pituitary | 626 |
| **Testing Total** | | **2,414** |
| **Grand Total** | | **12,064 images** |

### Processed Dataset (after setup_dataset.py)

#### Detection (tumor vs no_tumor)

| Split | Class | Count |
|---|---|---|
| Train | tumor | 7,705 |
| Train | no_tumor | 1,946 |
| Val | tumor | 963 |
| Val | no_tumor | 243 |
| Test | tumor | 964 |
| Test | no_tumor | 243 |
| **Detection Total** | | **12,064** |

#### Classification (glioma / meningioma / pituitary)

| Split | Class | Count |
|---|---|---|
| Train | glioma | 3,018 |
| Train | meningioma | 2,183 |
| Train | pituitary | 2,504 |
| Val | glioma | 377 |
| Val | meningioma | 273 |
| Val | pituitary | 313 |
| Test | glioma | 378 |
| Test | meningioma | 273 |
| Test | pituitary | 313 |
| **Classification Total** | | **9,632** |

> Note: Detection and Classification use the same source images organised differently — totals are not additive.

Source: [Brain Tumor MRI Dataset — Mendeley Data](https://data.mendeley.com/datasets/zwr4ntf94j/1)

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git LFS installed (`https://git-lfs.com`)
- GPU optional (CPU works for inference)

### 1. Clone

```bash
git clone https://github.com/vijay1845gs/brain_tumor_project.git
cd brain_tumor_project
git lfs pull   # downloads .pth weights + dataset images
```

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: http://localhost:3000

---

## Dataset Setup (if rebuilding from raw)

Raw `Training/` and `Testing/` folders are included via Git LFS.

```bash
cd backend
venv\Scripts\python.exe setup_dataset.py --src "C:\path\to\brain_tumor_project" --dst dataset
```

This creates:
```
dataset/
├── detection/
│   ├── train/  (tumor / no_tumor)
│   └── val/    (tumor / no_tumor)
└── classification/
    ├── train/  (glioma / meningioma / pituitary)
    └── val/    (glioma / meningioma / pituitary)
```

---

## Model Training

```bash
cd backend

# Train detection model (EfficientNet-B4)
venv\Scripts\python.exe train_detection.py --data dataset/detection --model efficientnet --epochs 20 --out models

# Train classification model (ResNet101)
venv\Scripts\python.exe train_classification.py --data dataset/classification --model resnet101 --epochs 25 --out models

# Train classification model (EfficientNet-B4)
venv\Scripts\python.exe train_classification_efficient.py --data dataset/classification --model efficientnet --epochs 15 --out models

```

Saved weights:
- `backend/models/detection_efficientnet.pth` — EfficientNet-B4 detection
- `backend/models/classification_model.pth` — ResNet101 classification

The server auto-loads weights on startup. Falls back to ImageNet pretrained if no weights found (demo mode).

---

## API Reference

### POST /predict/

Upload an MRI image and receive a full AI analysis.

**Request:** `multipart/form-data` with `file` field (JPEG/PNG/BMP, max 10MB)

**Response:**
```json
{
  "tumor_detected": true,
  "decision_type": "CONFIDENT",
  "tumor_type": "glioma",
  "confidence": 0.921,
  "uncertainty": 0.043,
  "reliability": "HIGH",
  "all_class_probs": { "glioma": 0.921, "meningioma": 0.054, "pituitary": 0.025 },
  "heatmap_image": "data:image/png;base64,...",
  "heatmap_gradcam": "data:image/png;base64,...",
  "heatmap_eigencam": "data:image/png;base64,...",
  "risk_level": "High",
  "risk_color": "red",
  "clinical_note": "...",
  "recommendation": "..."
}
```

---

### POST /report/

Generate a downloadable PDF clinical report from prediction data.

**Request:**
```json
{
  "user_name": "Dr. Smith",
  "tumor_detected": true,
  "tumor_type": "glioma",
  "confidence": 0.921,
  "uncertainty": 0.043,
  "reliability": "HIGH",
  "risk_level": "High",
  "clinical_note": "...",
  "recommendation": "...",
  "heatmap_image": "data:image/png;base64,..."
}
```

**Response:** `application/pdf` — `neuroscan_report.pdf`

---

### POST /analytics/summary

Compute statistics from a list of scan results (stateless — no DB).

**Request:**
```json
{
  "scans": [
    { "tumor_detected": true, "tumor_type": "glioma", "confidence": 0.92, "risk_level": "High" },
    { "tumor_detected": false, "tumor_type": null, "confidence": 0.11, "risk_level": "None" }
  ]
}
```

**Response:**
```json
{
  "total_scans": 2,
  "tumor_detected": 1,
  "no_tumor": 1,
  "detection_rate": 0.5,
  "average_confidence": 0.515,
  "tumor_type_distribution": { "glioma": 1 },
  "risk_distribution": { "High": 1, "None": 1 }
}
```

---

## Final Production Model Performance (Held-out Test Set)

Based on rigorous evaluation of completely unseen data (2,414 Test Set Images):

### Stage 1: Detection Ensemble (Tumor / No Tumor)
| Model | Accuracy | Recall (Sensitivity) | Precision | False Negatives |
|---|---|---|---|---|
| EfficientNet-B4 | 98.34% | 98.76% | 99.17% | 12 |
| ResNet-101 | 96.93% | 98.55% | 97.64% | 14 |
| **Ensemble (Deployed)** | **~ 99.0%** | **~ 99.2%** | **~ 99.4%** | **< 10** |

### Stage 2: Classification Ensemble (Glioma / Meningioma / Pituitary)
| Model | Accuracy | Weighted F1 | Precision | Recall |
|---|---|---|---|---|
| ResNet-101 | 97.20% | 97.14% | 96.99% | 97.35% |
| EfficientNet-B4 | 94.92% | 94.91% | 95.45% | 94.54% |
| **Ensemble (Deployed)** | **~ 97.8%** | **~ 97.8%** | **~ 97.5%** | **~ 98.0%** |



---

## Technologies

**Backend:** FastAPI 0.111, PyTorch 2.2, TorchVision 0.17, EfficientNet-B4, ResNet101, FPDF2 2.7.9, Pillow 10.3, OpenCV 4.9, NumPy 1.26, scikit-learn 1.4, Pydantic 2.7

**Frontend:** React 18.3, Vite, Tailwind CSS, Axios, Framer Motion, Lucide Icons

**AI Techniques:** Dual-Ensemble Inference (EfficientNet-B4 + ResNet101), Softmax Temperature Scaling (T=1.5), Grad-CAM++, EigenCAM, MC Dropout (5 passes), CLAHE, Skull Strip Simulation, TTA, FocalLoss, AdamW, WHO Risk Grading

**DevOps:** Docker, Git LFS, GitHub

---

## Git LFS

Large files are tracked via Git LFS:

| Pattern | Type | Size |
|---|---|---|
| `*.pth` | Model weights | ~250MB total |
| `Training/**` | Raw training images | ~185MB |
| `Testing/**` | Raw testing images | ~46MB |

After cloning, run `git lfs pull` to download all large files.

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.30` | Detection positive threshold |
| `LOW_THRESHOLD` | `0.15` | Detection negative threshold |
| `CLASSIFICATION_CONF_THRESHOLD` | `0.40` | Min confidence for confident classification |
| `MODEL_EFF_DET_PATH` | `models/detection_efficientnet.pth` | Primary Detection weights |
| `MODEL_RES_CLS_PATH` | `models/classification_model.pth` | Primary Classification weights |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `MAX_FILE_SIZE_MB` | `10` | Max upload size |

---

## License

MIT License

---

> **Medical Disclaimer:** This system is a research and educational tool. It must NOT be used as the sole basis for any clinical decision. Always consult a qualified medical professional.
