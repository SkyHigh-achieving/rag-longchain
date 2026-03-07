# Setup Script for Local Isolation
# This script creates a virtual environment inside the project folder.

Write-Host "--- Initializing Local RAG Environment ---" -ForegroundColor Cyan

# 1. Create Virtual Environment
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
}

# 2. Upgrade pip and install requirements
Write-Host "Installing dependencies..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Create necessary folders
$folders = @("models", "data", "vector_db")
foreach ($f in $folders) {
    if (!(Test-Path $f)) {
        New-Item -ItemType Directory -Path $f | Out-Null
        Write-Host "Created folder: $f"
    }
}

Write-Host "`nSetup Complete!" -ForegroundColor Green
Write-Host "To run the app, use: .\.venv\Scripts\python.exe app.py" -ForegroundColor Yellow
