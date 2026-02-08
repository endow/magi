$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot\frontend

if (!(Test-Path .env.local)) {
  Write-Host "frontend/.env.local not found. Copy frontend/.env.example to frontend/.env.local first." -ForegroundColor Yellow
  exit 1
}

if (!(Test-Path node_modules)) {
  Write-Host "Installing frontend dependencies..."
  npm install
}

Write-Host "Starting frontend on http://localhost:3000"
npm run dev
