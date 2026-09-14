$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$assetDirectory = Join-Path $projectRoot "src\gazetype\assets"
$modelPath = Join-Path $assetDirectory "face_landmarker.task"
$tempPath = "$modelPath.download"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
$minBytes = 3MB
$maxBytes = 10MB

New-Item -ItemType Directory -Path $assetDirectory -Force | Out-Null
try {
    Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $modelUrl -OutFile $tempPath -MaximumRedirection 3
    $size = (Get-Item $tempPath).Length
    if ($size -lt $minBytes -or $size -gt $maxBytes) {
        throw "Unexpected model size: $size bytes"
    }
    Move-Item -Path $tempPath -Destination $modelPath -Force
    Write-Host "Model indirildi ve doğrulandı: $modelPath ($size bytes)"
}
finally {
    Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
}
