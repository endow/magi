param(
  [switch]$NoReload
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot\backend

if (!(Test-Path .env)) {
  Write-Host "backend/.env not found. Copy backend/.env.example to backend/.env first." -ForegroundColor Yellow
  exit 1
}

$reloadArg = if ($NoReload) { "" } else { " --reload" }
$command = "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000$reloadArg"
Write-Host "Starting backend on http://localhost:8000"
Invoke-Expression $command
