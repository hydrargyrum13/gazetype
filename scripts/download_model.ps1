$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$assetDirectory = Join-Path $projectRoot "src\gazetype\assets"
$modelPath = Join-Path $assetDirectory "face_landmarker.task"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

New-Item -ItemType Directory -Path $assetDirectory -Force | Out-Null
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
Write-Host "Model indirildi: $modelPath"

