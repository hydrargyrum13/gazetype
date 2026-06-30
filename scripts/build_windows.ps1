$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$model = Join-Path $projectRoot "src\gazetype\assets\face_landmarker.task"

if (-not (Test-Path $python)) {
    throw ".venv bulunamadı. Önce Python 3.12 sanal ortamını oluşturun."
}

if (-not (Test-Path $model)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "download_model.ps1")
}

Push-Location $projectRoot
try {
    & $python -m PyInstaller --noconfirm --clean gazetype.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller başarısız oldu: $LASTEXITCODE" }
    Write-Host "Gazetype paketi hazır: $projectRoot\dist\Gazetype\Gazetype.exe"
}
finally {
    Pop-Location
}

