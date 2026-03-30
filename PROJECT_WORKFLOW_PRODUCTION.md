# NeuroScan AI: Complete Project Workflow (Production-Grade)

Last reviewed: 2026-03-24  
Repository root: `brain_tumor_project`

## 1. Scope and Method

This document is based on direct analysis of the project source/config files in:
- `backend/` (API, AI services, training pipeline, auth, analytics, history)
- `frontend/` (React UI, pages, API client, build/runtime config)
- root runtime files (`docker-compose.yml`, `start_project.ps1`, `README.md`, `explain.txt`, `.gitignore`)

The repository also contains large generated/runtime assets:
- `Training/`, `Testing/`, `backend/dataset/` (images)
- `frontend/node_modules/`, `backend/venv/`, `frontend/dist/` (dependencies/build artifacts)
- `backend/brain_tumor.db` (runtime database file)

These artifacts are part of repository state but not source-of-truth logic.

## 2. System Overview

NeuroScan AI is a full-stack MRI analysis platform with:
- FastAPI backend for inference and optional auth/history/analytics/user management modules
- PyTorch-based model inference pipeline (EfficientNet/ResNet, TTA, MC Dropout, CAM explainability)
- React + Vite frontend for upload, result visualization, analytics, and admin/profile flows
- SQLite persistence model for users and scan records
- Docker support for backend and frontend (Nginx static serving + reverse proxy)

## 3. Current Runtime Topology (As Implemented)

## 3.1 Backend API (active routes)

`backend/main.py` currently mounts only:
- `GET /`
- `GET /health`
- `POST /predict`
- `POST /predict/report`

Important: although modules for `/auth`, `/history`, `/analytics`, `/users` exist, they are not currently included in the FastAPI app router list.

## 3.2 Frontend router (active pages)

`frontend/src/App.jsx` currently mounts:
- `/` -> Landing page
- `/dashboard` -> MRI analysis page
- `/analytics` -> analytics page
- `*` -> Landing fallback

Important: pages/components for login/register/profile/history/admin/auth context/protected routes exist but are currently not wired into `App.jsx` + `main.jsx`.

## 4. End-to-End Workflow

## 4.1 User workflow (current live path)

1. User opens frontend.
2. User navigates to Dashboard.
3. User uploads MRI image (jpeg/png/bmp, max 10 MB).
4. Frontend calls `POST /predict` with multipart file.
5. Backend runs AI pipeline and returns:
   - tumor detection/classification
   - calibrated confidence
   - uncertainty + entropy
   - reliability label
   - risk summary + recommendation
   - Grad-CAM++ image
   - Score-CAM image
   - comparison strip
   - per-class probabilities and TTA agreement
6. Frontend renders detailed results panel and supports report download via `POST /predict/report`.

## 4.2 AI inference workflow (backend)

`routes/predict.py` -> `services/predictor.py`:

1. Validate MIME and file size.
2. Decode image bytes to PIL.
3. Preprocess:
   - resize
   - CLAHE
   - skull-strip simulation
   - normalize
   - generate TTA views
4. Load detection/classification models via singleton loader.
5. Detection stage:
   - TTA inference
   - MC Dropout sampling
   - confidence/uncertainty aggregation
6. Explainability stage:
   - Grad-CAM++
   - Score-CAM
   - optional comparison strip
7. If tumor detected:
   - classify tumor type
   - compute entropy/probability distribution
   - reliability decision rules
   - confidence-aware risk mapping
8. If no tumor:
   - return binary confidence + risk baseline
9. Return structured response to frontend.

## 4.3 PDF report workflow

1. Frontend posts prediction object to `POST /predict/report`.
2. Backend builds PDF using `fpdf2` with:
   - user label
   - status/type/confidence/uncertainty
   - heatmap image embedding
   - risk and recommendation text
   - medical disclaimer
3. Backend returns PDF bytes as download response.

## 4.4 Training workflow

`backend/train.py` supports:
- Tasks: `detection` or `classification`
- Models: `efficientnet` or `resnet101`
- Augmentations: MixUp, CutMix, heavy image transforms
- Losses: BCE/Focal/Label smoothing CE
- Sampler: WeightedRandomSampler
- Optimizer: AdamW with differential LR
- Scheduler: OneCycleLR
- Regularization: gradient clipping + early stopping
- Metrics: accuracy/precision/recall/F1/AUC/confusion matrix
- Artifacts:
  - `.pth` weights
  - training curves image
  - confusion matrix image
  - metrics JSON

## 4.5 Dataset preparation workflow

`backend/setup_dataset.py`:
1. Reads Kaggle dataset raw structure.
2. Builds task-specific structure:
   - detection: `tumor` vs `no_tumor`
   - classification: `glioma|meningioma|pituitary`
3. Uses `Training` as train and `Testing` as val split.

## 5. Data and Persistence Workflow

Intended persistence model:
- `users` table for auth/profile/admin
- `scan_records` table for inference history/feedback/analytics

Current behavior:
- ORM models exist.
- History/analytics/user routes exist.
- No active scan insert in prediction flow.
- `init_db()` is defined but not called in app startup.
- Result: history/analytics/user modules are present in code but not fully active in current running API path.

## 6. Deployment Workflow

## 6.1 Local script workflow (`start_project.ps1`)

1. Creates backend venv if missing.
2. Installs backend requirements.
3. Starts backend in separate PowerShell window.
4. Installs frontend deps if missing.
5. Starts frontend in separate PowerShell window.

## 6.2 Docker workflow (`docker-compose.yml`)

1. Build backend image from `backend/Dockerfile`.
2. Build frontend image from `frontend/Dockerfile`.
3. Run backend on `:8000`.
4. Run frontend Nginx on host `:3000`.
5. Frontend Nginx proxies `/auth` and `/predict` to backend.

Note: in container mode, `nginx.conf` does not proxy `/history`, `/analytics`, `/users` currently.

## 7. Production-Grade Readiness Assessment

## 7.1 Strengths already present

- Clear modular separation (routes/services/models/auth)
- Advanced inference logic (TTA + MC Dropout + dual CAM + entropy + risk mapping)
- Rich frontend visualization and reporting UX
- Dockerized backend/frontend
- Strong model training script for experimentation
- Typed schemas for many API contracts

## 7.2 Critical gaps before true production

P0 (must-fix):
- FastAPI app mounts only predict router; auth/history/analytics/users inactive.
- DB initialization not guaranteed at startup.
- Prediction results are not persisted to scan history table.
- Frontend routing does not include login/register/history/profile/admin pages.
- Auth provider/protected route infrastructure exists but is not wired in root app.
- Repository contains heavy runtime/dependency artifacts (`venv`, `node_modules`, datasets, `dist`, db file), increasing risk and operational drift.

P1 (high priority):
- Nginx proxy config incomplete for full API surface.
- No `.dockerignore`, resulting in oversized images/context.
- CORS is local-only hardcoded; no environment-driven production origin control.
- No test suite, CI pipeline, or deployment gating.
- No rate limiting, request tracing, structured logging, or metrics exporter.

P2 (next wave):
- Secrets hardening and rotation workflow.
- Role-based access and endpoint policy tests.
- SLOs, alerting, incident runbook automation.
- Model/version metadata and rollback controls.

## 8. Target Production Workflow (Recommended)

## 8.1 Request flow (target)

1. User authenticates (`/auth/login`) and receives JWT.
2. Protected dashboard upload calls `/predict`.
3. Backend predicts and writes scan record atomically.
4. User accesses:
   - `/history` for scan records and feedback
   - `/analytics/overview` for personal insights
   - `/users/me` for profile
5. Admin accesses:
   - `/users` + role management
   - `/analytics/admin/platform`
6. PDF generation is tied to persisted scan context for traceability.

## 8.2 CI/CD flow (target)

1. PR opened.
2. Automated checks:
   - backend lint/type/tests
   - frontend lint/build/tests
   - security/dependency scan
3. Build immutable images.
4. Deploy to staging; run smoke tests.
5. Manual approval gate.
6. Blue/green or rolling production deploy.
7. Post-deploy health + model smoke check.

## 8.3 Observability flow (target)

1. Structured logs with request IDs.
2. Metrics collection:
   - latency/error rates
   - model inference duration
   - confidence/uncertainty distributions
3. Alerting for:
   - rising 5xx
   - model timeout/failures
   - storage saturation
4. Dashboard for API + model + frontend error telemetry.

## 9. Minimal Implementation Plan to Reach Production-Ready Baseline

Phase 1: Activate full platform
- Mount all routers in `main.py`.
- Initialize DB in lifespan startup.
- Persist prediction output into `scan_records`.
- Wire frontend routes + auth context + protected routes.

Phase 2: Harden runtime
- Add `.dockerignore`.
- Remove committed runtime artifacts from repository history moving forward.
- Externalize CORS and runtime settings per environment.
- Expand Nginx proxy rules for complete backend API.

Phase 3: Quality gate
- Add backend unit/integration tests.
- Add frontend route/API integration tests.
- Add CI workflow with build + test + security scans.

Phase 4: Operations
- Add request logging, tracing, and metrics.
- Add alerts + on-call runbook.
- Add model/version metadata and rollback controls.

## 10. Environment and Config Inventory

Backend settings currently expected:
- `SECRET_KEY`
- `ALGORITHM`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `DATABASE_URL`
- `CONFIDENCE_THRESHOLD`
- `MODEL_DETECTION_PATH`
- `MODEL_CLASSIFICATION_PATH`

Runtime ports:
- Backend API: `8000`
- Frontend dev (Vite): `3000`
- Frontend Docker (Nginx mapped): `3000 -> 80`

## 11. Final Notes

This project has a strong technical base and already contains many advanced AI and UX capabilities.  
The key production step is not new features; it is wiring consistency and operational hardening so implemented modules run as one coherent platform.

