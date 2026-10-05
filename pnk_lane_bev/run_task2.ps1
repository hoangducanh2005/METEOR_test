param(
    [string]$Scene = "out/bevlane/pnk_295_1772351165_1772351179",
    [string]$Output = "out/task2_lane_bev/pnk_295_1772351165_1772351179",
    [string]$AllrootManifest = "out/allroot/pnk_295_1772351165_1772351179/run_manifest.json",
    [string]$Python = "pnk_comet_autolabel/.venv/Scripts/python.exe",
    [ValidateSet("auto", "cuda", "cpu")][string]$Device = "auto",
    [ValidateSet("lane-v2", "legacy")][string]$Pipeline = "lane-v2",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = Join-Path $RepoRoot $Python
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment not found: $PythonPath"
}

Push-Location $RepoRoot
try {
    $env:UV_CACHE_DIR = Join-Path $RepoRoot ".uv-cache"
    if ([System.IO.Path]::IsPathRooted($Output)) {
        $OutputAbsolute = [System.IO.Path]::GetFullPath($Output)
    }
    else {
        $OutputAbsolute = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Output))
    }
    $env:HF_HOME = Join-Path $OutputAbsolute "hf_cache"
    & uv pip install --python $PythonPath --no-deps --no-build-isolation -e (Join-Path $RepoRoot "pnk_lane_bev")
    if ($LASTEXITCODE -ne 0) { throw "Package installation failed" }
    $Checkpoint = Join-Path $RepoRoot "pnk_lane_bev/models/culane_res18.pth"
    $ClrnetCheckpoint = Join-Path $RepoRoot "pnk_lane_bev/models/culane_r18.pth"
    if (-not (Test-Path -LiteralPath $Checkpoint) -or -not (Test-Path -LiteralPath $ClrnetCheckpoint)) {
        & uv pip install --python $PythonPath "gdown>=5.2"
        if ($LASTEXITCODE -ne 0) { throw "gdown installation failed" }
        & $PythonPath (Join-Path $RepoRoot "pnk_lane_bev/download_models.py")
        if ($LASTEXITCODE -ne 0) { throw "Model download failed" }
    }
    $Command = if ($Pipeline -eq "lane-v2") { "lane-v2-run" } else { "run" }
    $Arguments = @(
        "-m", "pnk_lane_bev", $Command,
        "--scene", $Scene,
        "--output", $Output,
        "--checkpoint", $Checkpoint,
        "--ufld-checkpoint", $Checkpoint,
        "--clrnet-checkpoint", $ClrnetCheckpoint,
        "--allroot-manifest", $AllrootManifest,
        "--device", $Device
    )
    if ($Force) { $Arguments += "--force" }
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Task 2 pipeline failed" }
}
finally {
    Pop-Location
}
