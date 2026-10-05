param(
    [string]$Python = "pnk_comet_autolabel/.venv/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = Join-Path $RepoRoot $Python
Push-Location $RepoRoot
try {
    $env:UV_CACHE_DIR = Join-Path $RepoRoot ".uv-cache"
    & uv pip install --python $PythonPath "gdown>=5.2"
    if ($LASTEXITCODE -ne 0) { throw "gdown installation failed" }
    & $PythonPath (Join-Path $RepoRoot "pnk_lane_bev/download_models.py")
    if ($LASTEXITCODE -ne 0) { throw "UFLD-v2 checkpoint download failed" }
}
finally {
    Pop-Location
}
