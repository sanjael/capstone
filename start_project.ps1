# start_project.ps1
# NeuroScan AI v2.0 — Advanced Brain Tumor Detection System
# This script sets up and runs the complete project.

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   🧠 NeuroScan AI v2.0 — Starting System    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Backend Setup ──────────────────────────────────────────────────────────
Write-Host "📦 Setting up Backend..." -ForegroundColor Yellow
$BackendPath = Join-Path $Root "backend"
Set-Location $BackendPath

if (-not (Test-Path "venv")) {
    Write-Host "  Creating virtual environment..."
    python -m venv venv
}

Write-Host "  Activating venv and installing requirements..."
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
Write-Host "  ✅ Backend dependencies installed." -ForegroundColor Green

# Start Backend in new PowerShell window
$backendCmd = "Set-Location '$BackendPath'; .\venv\Scripts\activate; Write-Host '🧠 Backend starting on http://localhost:8000' -ForegroundColor Green; uvicorn main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Write-Host "  ⏳ Waiting 3s for backend to start..."
Start-Sleep -Seconds 3

# ── 2. Frontend Setup ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "📦 Setting up Frontend..." -ForegroundColor Yellow
$FrontendPath = Join-Path $Root "frontend"
Set-Location $FrontendPath

if (-not (Test-Path "node_modules")) {
    Write-Host "  Installing npm packages..."
    npm install --silent
}

Write-Host "  ✅ Frontend dependencies ready." -ForegroundColor Green

# Start Frontend in new PowerShell window
$frontendCmd = "Set-Location '$FrontendPath'; Write-Host '⚡ Frontend starting on http://localhost:5173' -ForegroundColor Green; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

# ── 3. Summary ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  🎉 NeuroScan AI is launching!               ║" -ForegroundColor Green
Write-Host "║                                              ║" -ForegroundColor Green
Write-Host "║  🌐 Landing Page:  http://localhost:5173     ║" -ForegroundColor Green
Write-Host "║  📊 Dashboard:     http://localhost:5173/dashboard  ║" -ForegroundColor Green
Write-Host "║  📖 API Docs:      http://localhost:8000/docs       ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Features available in v2.0:" -ForegroundColor Cyan
Write-Host "  ✅ AI Tumor Detection (ResNet101 + MC Dropout)"
Write-Host "  ✅ Grad-CAM++ Explainability Heatmaps"
Write-Host "  ✅ Scan History with Doctor Feedback"
Write-Host "  ✅ Analytics Dashboard"
Write-Host "  ✅ Clinical PDF Report Generation"
Write-Host "  ✅ User Profile Management"
Write-Host "  ✅ Admin Panel (User Management)"
Write-Host "  ✅ Beautiful Dark-Mode Landing Page"
Write-Host ""
